from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from numbers import Integral, Real
import warnings

import torch
from torch import nn
from torch.optim import Optimizer

from parsp.methods.utils import _get_layers


_ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
_FFN_PROJS = {"gate_proj", "up_proj", "down_proj"}
_REAL_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _compute_dtype(p):
    if p.dtype not in _REAL_DTYPES:
        raise TypeError(f"Unsupported parameter dtype: {p.dtype}")
    return torch.float64 if p.dtype == torch.float64 else torch.float32


def _layer_indices(indices, n_layers, label):
    result = set()
    for i in (() if indices is None else indices):
        if not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < n_layers:
            raise IndexError(f"Invalid {label} layer index: {i}")
        result.add(int(i))
    return result


def build_parameter_masks(
    model: nn.Module,
    top_attn_layer_idx=None,
    top_ffn_layer_idx=None,
    top_ffn_neurons=None,
    top_attn_neurons=None,
) -> dict[nn.Parameter, torch.Tensor]:
    
    layers = _get_layers(model)
    attn_ids = _layer_indices(top_attn_layer_idx, len(layers), "attention")
    ffn_ids = _layer_indices(top_ffn_layer_idx, len(layers), "FFN")
    masks = {}

    def group(neurons, selected, label):
        groups = defaultdict(set)
        for layer_idx, neuron_idx in (() if neurons is None else neurons):
            if (not isinstance(layer_idx, Integral) or isinstance(layer_idx, bool)
                    or int(layer_idx) not in selected):
                raise ValueError(f"{label} neuron belongs to an unselected layer: {layer_idx}")
            if (not isinstance(neuron_idx, Integral) or isinstance(neuron_idx, bool)
                    or neuron_idx < 0):
                raise IndexError(f"Invalid {label} neuron index: {neuron_idx}")
            groups[int(layer_idx)].add(int(neuron_idx))
        return groups

    def add(module, indices, axis, name, allow_partial=False):
        weight = module.weight
        if weight.ndim != 2 or not weight.requires_grad:
            raise ValueError(f"{name}: expected a trainable matrix; call layer_division first.")
        size = weight.shape[axis]
        valid = sorted(i for i in indices if i < size)
        if len(valid) != len(indices):
            if not allow_partial:
                raise IndexError(f"{name}: neuron indices exceed axis {axis} of size {size}")
            warnings.warn(f"{name}: legacy V mapping discarded {len(indices)-len(valid)} indices.",
                          RuntimeWarning, stacklevel=2)
        if not valid:
            warnings.warn(f"{name}: no valid indices; preserving legacy behavior (no mask).",
                          RuntimeWarning, stacklevel=2)
            return
        mask = torch.zeros(size, device=weight.device, dtype=torch.bool)
        mask[valid] = True
        if weight in masks:
            raise ValueError(f"{name}: shared weight has multiple mask definitions.")
        masks[weight] = mask.reshape(-1, 1) if axis == 0 else mask.reshape(1, -1)

    for i, indices in sorted(group(top_ffn_neurons, ffn_ids, "FFN").items()):
        mlp = layers[i].mlp
        add(mlp.gate_proj, indices, 0, f"layers.{i}.mlp.gate_proj")
        add(mlp.up_proj, indices, 0, f"layers.{i}.mlp.up_proj")
        add(mlp.down_proj, indices, 1, f"layers.{i}.mlp.down_proj")

    for i, indices in sorted(group(top_attn_neurons, attn_ids, "attention").items()):
        attn = layers[i].self_attn
        # Validate O-space indices before applying the legacy V filter.
        if max(indices) >= attn.o_proj.weight.shape[1]:
            raise IndexError(f"layers.{i}.self_attn.o_proj: invalid neuron index")
        if attn.v_proj.weight.shape[0] != attn.o_proj.weight.shape[1]:
            warnings.warn(f"Layer {i}: V/O dimensions differ; using legacy index filtering, not GQA mapping.",
                          RuntimeWarning, stacklevel=2)
        add(attn.v_proj, indices, 0, f"layers.{i}.self_attn.v_proj", allow_partial=True)
        add(attn.o_proj, indices, 1, f"layers.{i}.self_attn.o_proj")
    return masks


@torch.no_grad()
def build_parameter_projectors(
    model: nn.Module,
    top_attn_layer_idx=None,
    top_ffn_layer_idx=None,
    rcond: float = 1e-6,
    min_ratio: float = 0.3,
    dataset_name: str | None = None,
    dataset_size: int | None = None,
    seed: int | None = None,
    cache_dir: str | None = None,
    *,
    projector_builder: Callable,
    use_cache: bool = False,
    release_covariance: bool = True,
    projector_device: str | torch.device | None = None,
) -> dict[nn.Parameter, torch.Tensor]:

    layers = _get_layers(model)
    attn_ids = _layer_indices(top_attn_layer_idx, len(layers), "attention")
    ffn_ids = _layer_indices(top_ffn_layer_idx, len(layers), "FFN")
    config = model.config
    model_type = getattr(config, "model_type", "").lower()
    model_id = getattr(config, "_name_or_path", None)
    if use_cache and (not model_id or dataset_name is None or dataset_size is None or seed is None):
        raise ValueError("Projector caching requires model ID, dataset_name, dataset_size and seed.")
    if not callable(projector_builder):
        raise TypeError("projector_builder must be the existing projector construction function.")
    if not math.isfinite(rcond) or rcond < 0 or not 0 <= min_ratio <= 1:
        raise ValueError("Invalid rcond or min_ratio.")
    dataset_tag = f"{dataset_name}_size_{dataset_size}_seed_{seed}"
    names = {id(module): name for name, module in model.named_modules()}
    targets = []
    for i in sorted(attn_ids | ffn_ids):
        layer = layers[i]
        for selected, block, allowed in ((i in attn_ids, "self_attn", _ATTN_PROJS),
                                         (i in ffn_ids, "mlp", _FFN_PROJS)):
            if not selected:
                continue
            for proj_name in sorted(allowed):
                module = getattr(getattr(layer, block), proj_name)
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"Layer {i}.{block}.{proj_name}: expected nn.Linear.")
                if not module.weight.requires_grad:
                    continue
                name = names[id(module)]
                dim = module.weight.shape[1]
                cov = getattr(module, "covariance_matrix", None)
                if not isinstance(cov, torch.Tensor) or cov.shape != (dim, dim):
                    raise ValueError(f"{name}: missing or invalid covariance_matrix; expected {(dim, dim)}.")
                targets.append((name, module))

    projectors = {}

    for name, module in targets:
        p = module.weight
        if p in projectors:
            raise ValueError(f"{name}: shared weight has multiple projector definitions.")
        cache_name = name
        if model_type.startswith("qwen") and name.startswith("model.layers."):
            cache_name = name[len("model."):]
        P = projector_builder(
            module.covariance_matrix,
            rcond=rcond,
            min_ratio=min_ratio,
            use_cache=use_cache,
            cache_dir=cache_dir,
            layer_name=cache_name,
            model_id=model_id,
            dataset_tag=dataset_tag,
        )
        n = p.shape[1]
        if (not isinstance(P, torch.Tensor) or P.shape != (n, n)
                or not P.is_floating_point() or not torch.isfinite(P).all().item()):
            raise ValueError(f"{name}: invalid projector; expected a finite real {(n, n)} tensor.")
        projectors[p] = P.detach().to(device=p.device if projector_device is None else projector_device,
                                      dtype=_compute_dtype(p))
    if release_covariance:
        for _, module in targets:
            delattr(module, "covariance_matrix")
    return projectors


def constrain_update(delta, mask=None, projector=None):

    if mask is not None:
        delta = delta * mask.to(device=delta.device, dtype=delta.dtype)
    if projector is not None:
        delta = delta @ projector.to(device=delta.device, dtype=delta.dtype)
    return delta


class PaRSPAdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, *, masks=None, projectors=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._validate_group(defaults)
        super().__init__(params, defaults)
        self.set_constraints(masks=masks, projectors=projectors)

    @staticmethod
    def _validate_group(group):
        for name in ("lr", "eps", "weight_decay"):
            value = group[name]
            if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar.")
        betas = group["betas"]
        if len(betas) != 2 or any(not isinstance(b, Real) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas must contain two scalars in [0, 1).")
        for flag in ("amsgrad", "maximize", "foreach", "fused", "capturable", "differentiable"):
            if group.get(flag, False):
                raise ValueError(f"PaRSPAdamW does not support {flag}=True.")

    def add_param_group(self, param_group):
        self._validate_group({**self.defaults, **param_group})
        super().add_param_group(param_group)

    def set_constraints(self, *, masks=None, projectors=None):

        params = {id(p): p for g in self.param_groups for p in g["params"]}
        new_masks, new_projectors = {}, {}
        for mapping, dest, kind in ((masks, new_masks, "mask"),
                                     (projectors, new_projectors, "projector")):
            if mapping is None:
                continue
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{kind}s must be a mapping from Parameter to Tensor.")
            for p, tensor in mapping.items():
                if not isinstance(p, nn.Parameter) or id(p) not in params:
                    raise ValueError(f"{kind} refers to a parameter outside this optimizer.")
                if p.ndim != 2 or not p.requires_grad:
                    raise ValueError(f"{kind} requires a trainable two-dimensional parameter.")
                if not isinstance(tensor, torch.Tensor) or tensor.is_complex():
                    raise TypeError(f"{kind} must be a real tensor.")
                m, n = p.shape
                if kind == "mask":
                    if tensor.shape not in ((m, 1), (1, n), (n,), (m, n)):
                        raise ValueError(f"Invalid mask shape {tuple(tensor.shape)} for {(m,n)}.")
                    if not ((tensor == 0) | (tensor == 1)).all().item():
                        raise ValueError("Mask must be binary.")
                    dest[p] = tensor.detach().to(device=p.device, dtype=torch.bool)
                else:
                    if tensor.shape != (n, n) or not tensor.is_floating_point():
                        raise ValueError(f"Projector must be a real floating-point {(n,n)} tensor.")
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError("Projector contains NaN or Inf.")
                    dest[p] = tensor.detach().to(dtype=_compute_dtype(p))
        self.masks, self.projectors = new_masks, new_projectors

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._validate_group(group)
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                _compute_dtype(p)
                if p.layout != torch.strided or p.grad.layout != torch.strided:
                    raise RuntimeError("PaRSPAdamW supports dense strided parameters and gradients only.")

        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                dtype = _compute_dtype(p)
                grad = p.grad.detach().to(dtype=dtype)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                del grad
                bc1, bc2 = 1 - beta1 ** t, 1 - beta2 ** t
                denom = v.sqrt().div_(math.sqrt(bc2)).add_(eps)
                delta = m.div(denom).mul_(-lr / bc1)
                del denom
                if wd != 0:
                    delta.add_(p.detach(), alpha=-lr * wd)
                delta = constrain_update(delta, self.masks.get(p), self.projectors.get(p))
                delta.add_(p.detach())
                p.copy_(delta)
                del delta
        return loss

    def load_state_dict(self, state_dict):
        saved_groups = state_dict["param_groups"]
        saved_states = state_dict["state"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("Optimizer checkpoint parameter group count mismatch.")
        restore = []
        for old_group, new_group in zip(saved_groups, self.param_groups):
            self._validate_group({**self.defaults, **old_group})
            if len(old_group["params"]) != len(new_group["params"]):
                raise ValueError("Optimizer checkpoint parameter group size mismatch.")
            for old_id, p in zip(old_group["params"], new_group["params"]):
                st = saved_states.get(old_id, {})
                if not st:
                    continue
                step = st["step"]
                step = step.item() if isinstance(step, torch.Tensor) else step
                if not isinstance(step, Real) or not math.isfinite(step) or step < 0 or int(step) != step:
                    raise ValueError("Invalid optimizer step in checkpoint.")
                for key in ("exp_avg", "exp_avg_sq"):
                    if not isinstance(st[key], torch.Tensor) or st[key].shape != p.shape:
                        raise ValueError(f"Invalid checkpoint state {key}.")
                restore.append((p, int(step), st["exp_avg"], st["exp_avg_sq"]))
        super().load_state_dict(state_dict)
        for p, step, m, v in restore:
            dtype = _compute_dtype(p)
            self.state[p]["step"] = step
            self.state[p]["exp_avg"] = m.detach().to(device=p.device, dtype=dtype, copy=True)
            self.state[p]["exp_avg_sq"] = v.detach().to(device=p.device, dtype=dtype, copy=True)
