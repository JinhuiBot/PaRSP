import os
import gc
import torch
import torch.nn as nn

from tqdm import tqdm
from typing import List


def common_null_space_projector_sum_method(
        total_cov: torch.Tensor,
        rcond: float,
        min_ratio: float,
        use_cache: bool = False,
        cache_dir: str = "./cache_projectors",
        layer_name: str = None,
        model_id: str = None,
        dataset_tag: str = None
):

    if not isinstance(total_cov, torch.Tensor):
        raise ValueError(
            f"Expected torch.Tensor, got {type(total_cov)}"
        )

    if total_cov.ndim != 2 or total_cov.shape[0] != total_cov.shape[1]:
        raise ValueError(
            f"Expected a square covariance matrix, "
            f"got {tuple(total_cov.shape)}"
        )

    if total_cov.shape[0] == 0:
        raise ValueError("Covariance matrix is empty.")

    if rcond < 0 or not 0 <= min_ratio <= 1:
        raise ValueError("Invalid rcond or min_ratio.")

    cache_dir = os.path.expanduser(
        cache_dir or "./cache_projectors"
    )

    cache_path = None

    if use_cache:
        if not all([layer_name, model_id, dataset_tag]):
            print(
                f"[Warning] Missing cache metadata for "
                f"{layer_name}. Skipping cache."
            )
        else:
            os.makedirs(cache_dir, exist_ok=True)

            safe_model = model_id.replace("/", "_")
            safe_rcond = str(rcond).replace(".", "p").replace("-", "n")
            safe_ratio = str(min_ratio).replace(".", "p")

            fname = (
                f"P_{safe_model}_{dataset_tag}_{layer_name}"
                f"_rcond_{safe_rcond}_minratio_{safe_ratio}.pt"
            )

            cache_path = os.path.join(cache_dir, fname)

            if os.path.exists(cache_path):
                try:
                    P = torch.load(
                        cache_path,
                        map_location="cpu",
                        weights_only=True
                    )

                    if (
                        isinstance(P, torch.Tensor)
                        and P.shape == total_cov.shape
                        and P.dtype == torch.float32
                        and torch.isfinite(P).all()
                    ):
                        return P

                    print(
                        f"[Warning] Invalid cached projector: "
                        f"{cache_path}. Recomputing..."
                    )

                except Exception as e:
                    print(
                        f"[Warning] Failed to load cache "
                        f"{cache_path}: {e}. Recomputing..."
                    )

    device = torch.device("cpu")
    dtype = torch.float32

    matrix = total_cov.detach().to(
        device=device,
        dtype=dtype
    )

    m = matrix.shape[0]

    if not torch.isfinite(matrix).all():
        raise ValueError(
            f"Non-finite covariance matrix for {layer_name}."
        )

    try:
        S, U = torch.linalg.eigh(matrix)

    except RuntimeError:
        jitter = 1e-6 * torch.eye(
            m,
            device=device,
            dtype=dtype
        )
        S, U = torch.linalg.eigh(matrix + jitter)

    max_eig = S[-1]
    tol = rcond * max_eig
    null_mask = S <= tol

    num_null_vectors = null_mask.sum().item()
    total_vectors = S.shape[0]
    current_ratio = num_null_vectors / total_vectors

    min_required_count = int(total_vectors * min_ratio)

    forced_by_ratio = num_null_vectors < min_required_count

    if forced_by_ratio:
        null_mask = torch.zeros_like(S, dtype=torch.bool)
        null_mask[:min_required_count] = True
        num_null_vectors = min_required_count

    final_ratio = num_null_vectors / total_vectors

    info_msg = (
        f"[Info] {layer_name or 'Unknown'}: "
        f"Null Dim {num_null_vectors}/{total_vectors} "
        f"({final_ratio:.2%})"
    )

    if forced_by_ratio:
        print(
            f"{info_msg} | [Force Override] "
            f"{current_ratio:.2%} < {min_ratio:.2%}"
        )
    else:
        print(
            f"{info_msg} | [By Threshold] "
            f"{current_ratio:.2%} >= {min_ratio:.2%}"
        )

    Z = U[:, null_mask]

    if Z.shape[1] == 0:
        P = torch.zeros(
            (m, m),
            device=device,
            dtype=dtype
        )

        print(
            f"[Warning] {layer_name}: "
            f"Null space is empty! P is all zeros."
        )
    else:
        P = Z @ Z.T

    if use_cache and cache_path is not None:
        try:
            torch.save(P, cache_path)

        except Exception as e:
            print(
                f"[Warning] Failed to save cache "
                f"to {cache_path}: {e}"
            )

    return P


def register_nullspace_hooks(
        model: nn.Module,
        top_attn_layer_idx: List[int],
        top_ffn_layer_idx: List[int],
        rcond: float,
        min_ratio: float,
        dataset_name: str = None,
        dataset_size: int = None,
        seed: int = None,
        cache_dir: str = None,
        *,
        use_cache: bool = True,
        strict: bool = False
) -> List[torch.utils.hooks.RemovableHandle]:

    attn_layers = set(top_attn_layer_idx)
    ffn_layers = set(top_ffn_layer_idx)
    target_layers = attn_layers | ffn_layers

    SUPPORTED_ATTN_PROJS = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    }

    SUPPORTED_FFN_PROJS = {
        "gate_proj",
        "up_proj",
        "down_proj"
    }

    handles = []
    modules_to_process = []

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "").lower()
    model_id = getattr(config, "_name_or_path", None)

    print("Scanning modules for hook registration...")

    for name, module in model.named_modules():

        if not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")

        try:
            layer_pos = parts.index("layers")
            layer_idx = int(parts[layer_pos + 1])

        except (ValueError, IndexError):
            continue

        if layer_idx not in target_layers:
            continue

        submodule_name = parts[-1]

        is_attn = any(
            p in {"self_attn", "attn", "attention"}
            for p in parts
        )

        is_ffn = any(
            p in {"mlp", "feed_forward"}
            for p in parts
        )

        should_process = (
            (
                is_attn
                and layer_idx in attn_layers
                and submodule_name in SUPPORTED_ATTN_PROJS
            )
            or
            (
                is_ffn
                and layer_idx in ffn_layers
                and submodule_name in SUPPORTED_FFN_PROJS
            )
        )

        if should_process:
            modules_to_process.append(
                (name, module)
            )

    print(
        f"Found {len(modules_to_process)} target modules. "
        f"Starting calculation..."
    )

    dataset_tag = (
        f"{dataset_name}_size_{dataset_size}_seed_{seed}"
    )

    cache_enabled = (
        use_cache
        and bool(model_id)
        and dataset_name is not None
        and dataset_size is not None
        and seed is not None
    )

    def make_hook(projector):
        def hook(grad):
            projector_local = projector.to(
                device=grad.device,
                dtype=grad.dtype
            )
            return grad @ projector_local

        return hook

    for i, (name, module) in tqdm(
        enumerate(modules_to_process),
        total=len(modules_to_process)
    ):
        if not module.weight.requires_grad:
            print(f"[Skip] Frozen weight: {name}")
            continue

        try:
            if not hasattr(module, "covariance_matrix"):
                raise AttributeError(
                    f"Missing covariance_matrix for {name}"
                )

            cov_tensor = module.covariance_matrix

            if not isinstance(cov_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected Tensor but got "
                    f"{type(cov_tensor)} in {name}"
                )

            expected_dim = module.weight.shape[1]

            if cov_tensor.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Covariance shape mismatch in {name}: "
                    f"expected {(expected_dim, expected_dim)}, "
                    f"got {tuple(cov_tensor.shape)}"
                )

            cache_layer_name = name

            if (
                model_type.startswith("qwen")
                and name.startswith("model.layers.")
            ):
                cache_layer_name = name[len("model."):]

            P_cpu = common_null_space_projector_sum_method(
                cov_tensor,
                rcond=rcond,
                min_ratio=min_ratio,
                use_cache=cache_enabled,
                cache_dir=cache_dir,
                layer_name=cache_layer_name,
                model_id=model_id,
                dataset_tag=dataset_tag
            )

            P_device = P_cpu.to(
                device=module.weight.device,
                dtype=module.weight.dtype
            )

            del P_cpu

            handle = module.weight.register_hook(
                make_hook(P_device)
            )

            handles.append(handle)

            delattr(module, "covariance_matrix")
            del cov_tensor

        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")

            if strict:
                for handle in handles:
                    handle.remove()
                handles.clear()
                raise

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        if (i + 1) % 10 == 0:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Registered {len(handles)} null-space hooks."
    )

    return handles