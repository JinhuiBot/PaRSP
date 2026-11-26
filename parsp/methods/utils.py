import torch


def _unwrap_model(model):
    if hasattr(model, "captured_data") and hasattr(model, "model"):
        return model.model
    return model


def _get_layers(model):
    hf_model = _unwrap_model(model)

    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers

    if hasattr(hf_model, "layers"):
        return hf_model.layers

    raise AttributeError("Cannot find Transformer layers in model.")


def transfer_output(captured_data, n_layers):

    if not isinstance(captured_data, dict) or len(captured_data) == 0:
        return tuple(
            [[] for _ in range(n_layers)]
            for _ in range(8)
        )

    all_pos_layer_input = []
    all_pos_attn_output = []
    all_pos_residual_output = []
    all_pos_ffn_output = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []
    all_pos_coefficient_scores = []
    all_attn_scores = []

    def get_last_token(tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor

        tensor = tensor.detach().float().cpu()

        if tensor.dim() == 3:
            return tensor[:, -1, :].numpy()

        if tensor.dim() == 2:
            return tensor[-1, :].unsqueeze(0).numpy()

        return tensor.numpy()

    for i in range(n_layers):
        if i not in captured_data:
            raise KeyError(
                f"Missing captured data for layer {i}."
            )

        layer_data = captured_data[i]

        inp = layer_data.get("layer_input")
        attn = layer_data.get("attn_output")

        all_pos_layer_input.append(
            get_last_token(inp)
        )

        all_pos_attn_output.append(
            get_last_token(attn)
        )

        if inp is not None and attn is not None:
            all_pos_residual_output.append(
                get_last_token(inp + attn)
            )
        else:
            all_pos_residual_output.append(None)

        all_pos_ffn_output.append(
            get_last_token(layer_data.get("ffn_output"))
        )

        all_pos_layer_output.append(
            get_last_token(layer_data.get("layer_output"))
        )

        all_last_attn_subvalues.append(
            get_last_token(layer_data.get("attn_subvalues"))
        )

        all_pos_coefficient_scores.append(
            get_last_token(layer_data.get("ffn_inter"))
        )

        all_attn_scores.append(None)

    return (
        all_pos_layer_input,
        all_pos_attn_output,
        all_pos_residual_output,
        all_pos_ffn_output,
        all_pos_layer_output,
        all_last_attn_subvalues,
        all_pos_coefficient_scores,
        all_attn_scores
    )


def get_bsvalues(hidden_states, model, final_var=None):

    hf_model = _unwrap_model(model)

    if not hasattr(hf_model, "lm_head"):
        raise AttributeError("Cannot find lm_head in model.")

    lm_head = hf_model.lm_head
    head_device = lm_head.weight.device
    head_dtype = lm_head.weight.dtype

    if hasattr(hf_model, "model") and hasattr(hf_model.model, "norm"):
        norm_module = hf_model.model.norm
    elif hasattr(hf_model, "norm"):
        norm_module = hf_model.norm
    else:
        raise AttributeError("Cannot find final normalization module.")

    norm_device = norm_module.weight.device

    hidden_states = torch.as_tensor(
        hidden_states,
        device=norm_device,
        dtype=torch.float32
    )

    if final_var is not None:

        final_var = torch.as_tensor(
            final_var,
            device=norm_device,
            dtype=torch.float32
        )

        if final_var.ndim > 0 and final_var.shape[0] != 1:
            final_var = final_var[-1]
        if final_var.numel() != 1:
            raise ValueError(
                f"Expected a scalar normalization reference, "
                f"got shape {tuple(final_var.shape)}."
            )

        final_var = final_var.reshape(())

        model_type = getattr(
            hf_model.config, "model_type", ""
        ).lower()

        if model_type.startswith("qwen"):
            denominator = final_var + 1e-6

        elif model_type == "llama":
            denominator = torch.clamp(final_var, min=1e-6)

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}"
            )

        norm_weight = norm_module.weight.detach().to(
            device=norm_device,
            dtype=torch.float32
        )

        hidden_states_norm = (
            hidden_states / denominator
        ) * norm_weight

    else:
        hidden_states_norm = norm_module(hidden_states)

    hidden_states_norm = hidden_states_norm.to(
        device=head_device,
        dtype=head_dtype
    )

    logits = lm_head(hidden_states_norm)

    return logits.float()


def get_fc2_params(model, layer_idx):

    layers = _get_layers(model)

    if not 0 <= layer_idx < len(layers):
        raise IndexError(
            f"Layer index {layer_idx} is out of range."
        )

    return layers[layer_idx].mlp.down_proj.weight.detach()