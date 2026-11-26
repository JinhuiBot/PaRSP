from collections import defaultdict
import warnings
from parsp.methods.utils import _get_layers
import torch


def layer_division(
        model,
        top_attn_layer_idx=None,
        top_ffn_layer_idx=None
):

    layers = _get_layers(model)
    n_layers = len(layers)

    attn_indices = (
        set(top_attn_layer_idx)
        if top_attn_layer_idx is not None
        else set()
    )

    ffn_indices = (
        set(top_ffn_layer_idx)
        if top_ffn_layer_idx is not None
        else set()
    )

    for idx in attn_indices:
        if not 0 <= idx < n_layers:
            raise IndexError(
                f"Attention layer index {idx} is out of range "
                f"[0, {n_layers - 1}]."
            )

    for idx in ffn_indices:
        if not 0 <= idx < n_layers:
            raise IndexError(
                f"FFN layer index {idx} is out of range "
                f"[0, {n_layers - 1}]."
            )

    for param in model.parameters():
        param.requires_grad_(False)

    for idx in sorted(attn_indices):
        for param in layers[idx].self_attn.parameters():
            param.requires_grad_(True)

    for idx in sorted(ffn_indices):
        for param in layers[idx].mlp.parameters():
            param.requires_grad_(True)

    print(
        f"Layer division completed. "
        f"Attention: {len(attn_indices)}, "
        f"FFN: {len(ffn_indices)}."
    )


def neuron_division(
        model,
        top_attn_layer_idx=None,
        top_ffn_layer_idx=None,
        top_ffn_neurons=None,
        top_attn_neurons=None
):

    layers = _get_layers(model)

    attn_indices = (
        set(top_attn_layer_idx)
        if top_attn_layer_idx is not None
        else set()
    )

    ffn_indices = (
        set(top_ffn_layer_idx)
        if top_ffn_layer_idx is not None
        else set()
    )

    handles = []

    def group_neurons(neurons, selected_layers, group_name):
        neuron_map = defaultdict(set)

        if neurons is None:
            return neuron_map

        for layer_idx, neuron_idx in neurons:

            if not 0 <= layer_idx < len(layers):
                raise IndexError(
                    f"Invalid {group_name} layer index: {layer_idx}."
                )

            if layer_idx not in selected_layers:
                raise ValueError(
                    f"{group_name} neuron selected in "
                    f"non-trainable layer {layer_idx}."
                )

            if neuron_idx < 0:
                raise IndexError(
                    f"Invalid {group_name} neuron index: {neuron_idx}."
                )

            neuron_map[layer_idx].add(neuron_idx)

        return neuron_map

    ffn_map = group_neurons(
        top_ffn_neurons,
        ffn_indices,
        "FFN"
    )

    attn_map = group_neurons(
        top_attn_neurons,
        attn_indices,
        "Attention"
    )

    layer_division(
        model,
        top_attn_layer_idx=attn_indices,
        top_ffn_layer_idx=ffn_indices
    )

    def apply_mask_hook(
            module,
            indices,
            axis,
            allow_partial=False
    ):
        weight = module.weight

        if axis not in (0, 1):
            raise ValueError(f"Unsupported mask axis: {axis}.")

        if not weight.requires_grad:
            raise RuntimeError(
                "Cannot register mask hook on a frozen weight."
            )

        dim_size = weight.shape[axis]

        indices = sorted(set(indices))

        valid_indices = [
            idx for idx in indices
            if 0 <= idx < dim_size
        ]

        if len(valid_indices) != len(indices):

            if not allow_partial:
                raise IndexError(
                    f"Neuron index out of range for "
                    f"weight shape {tuple(weight.shape)}, "
                    f"axis={axis}."
                )

            warnings.warn(
                f"Filtered {len(indices) - len(valid_indices)} "
                f"out-of-range neuron indices for "
                f"weight shape {tuple(weight.shape)}, "
                f"axis={axis}.",
                RuntimeWarning
            )

        if not valid_indices:
            warnings.warn(
                "No valid neuron indices. "
                "Skipping mask registration to preserve "
                "the legacy PaRSP behavior.",
                RuntimeWarning
            )
            return

        mask = torch.zeros(
            dim_size,
            device=weight.device,
            dtype=weight.dtype
        )

        mask[valid_indices] = 1.0

        if axis == 0:
            mask = mask.view(-1, 1)
        else:
            mask = mask.view(1, -1)

        def hook_fn(grad):
            return grad * mask.to(
                device=grad.device,
                dtype=grad.dtype
            )

        handle = weight.register_hook(hook_fn)
        handles.append(handle)

    try:
        for layer_idx, indices in sorted(ffn_map.items()):

            mlp = layers[layer_idx].mlp

            apply_mask_hook(
                mlp.gate_proj,
                indices,
                axis=0
            )

            apply_mask_hook(
                mlp.up_proj,
                indices,
                axis=0
            )

            apply_mask_hook(
                mlp.down_proj,
                indices,
                axis=1
            )

        for layer_idx, indices in sorted(attn_map.items()):

            attn = layers[layer_idx].self_attn

            if attn.v_proj.weight.shape[0] != attn.o_proj.weight.shape[1]:
                warnings.warn(
                    f"Attention dimension mismatch in layer "
                    f"{layer_idx}. Legacy index filtering is used; ",
                    RuntimeWarning
                )

            apply_mask_hook(
                attn.v_proj,
                indices,
                axis=0,
                allow_partial=True
            )

            apply_mask_hook(
                attn.o_proj,
                indices,
                axis=1
            )

    except Exception:
        for handle in handles:
            handle.remove()
        raise

    num_ffn_neurons = sum(
        len(indices) for indices in ffn_map.values()
    )

    num_attn_neurons = sum(
        len(indices) for indices in attn_map.values()
    )

    print(
        f"Neuron division completed. "
        f"FFN neurons: {num_ffn_neurons}, "
        f"Attention neurons: {num_attn_neurons}, "
        f"Registered hooks: {len(handles)}."
    )

    return handles