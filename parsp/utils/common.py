import functools
import gc
import warnings
from typing import Optional

import torch


def get_prob(logits):
    return torch.softmax(logits, dim=-1)


def get_log_prob(logits):
    return torch.log_softmax(logits, dim=-1)


def cuda_memory_cleanup(
        threshold_mb: Optional[float] = None,
        debug: bool = False,
        *,
        device=None,
        use_reserved: bool = False,
        collect_ipc: bool = False
):

    if threshold_mb is not None and threshold_mb < 0:
        raise ValueError("threshold_mb must be non-negative.")

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            if not torch.cuda.is_available():
                return func(*args, **kwargs)

            cuda_device = (
                torch.device("cuda", torch.cuda.current_device())
                if device is None
                else torch.device(device)
            )

            if cuda_device.type != "cuda":
                return func(*args, **kwargs)

            try:
                return func(*args, **kwargs)

            finally:
                try:
                    with torch.cuda.device(cuda_device):

                        mb = 1024 ** 2

                        current_allocated = (
                            torch.cuda.memory_allocated(cuda_device)
                        )

                        current_reserved = (
                            torch.cuda.memory_reserved(cuda_device)
                        )

                        monitored_memory = (
                            current_reserved
                            if use_reserved
                            else current_allocated
                        )

                        should_clean = (
                            threshold_mb is None
                            or monitored_memory / mb > threshold_mb
                        )

                        if should_clean:

                            if debug:
                                print(
                                    f"Before cleanup [{cuda_device}] - "
                                    f"Allocated: "
                                    f"{current_allocated / mb:.2f} MB, "
                                    f"Reserved: "
                                    f"{current_reserved / mb:.2f} MB"
                                )

                            gc.collect()

                            torch.cuda.empty_cache()

                            if collect_ipc:
                                torch.cuda.ipc_collect()

                            if debug:

                                final_allocated = (
                                    torch.cuda.memory_allocated(
                                        cuda_device
                                    )
                                )

                                final_reserved = (
                                    torch.cuda.memory_reserved(
                                        cuda_device
                                    )
                                )

                                cleaned_allocated = (
                                    current_allocated
                                    - final_allocated
                                ) / mb

                                cleaned_reserved = (
                                    current_reserved
                                    - final_reserved
                                ) / mb

                                print(
                                    f"After cleanup [{cuda_device}] - "
                                    f"Allocated: "
                                    f"{final_allocated / mb:.2f} MB, "
                                    f"Reserved: "
                                    f"{final_reserved / mb:.2f} MB"
                                )

                                print(
                                    f"Cleaned - "
                                    f"Allocated: "
                                    f"{cleaned_allocated:.2f} MB, "
                                    f"Reserved: "
                                    f"{cleaned_reserved:.2f} MB"
                                )

                except Exception as exc:
                    warnings.warn(
                        f"CUDA memory cleanup failed: {exc}",
                        RuntimeWarning
                    )

        return wrapper

    return decorator