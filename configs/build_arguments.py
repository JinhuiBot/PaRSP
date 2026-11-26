from dataclasses import dataclass, field
from typing import List


@dataclass
class BuildArguments:
    model_id: str = field(default="Llama-3.1-8B")
    device: str = field(default="cuda:0")
    layer_ratio: float
    neuron_ratio: float
    rcond: float = field
    min_ratio: float = field
    dataset_name: str = field(default=None)
    data_path: str = field(default=None)
    cov_dataset_name: str = field(default=None)
    cov_data_path: str = field(default=None)
    train_size: int = field(default=1000)
    test_size: int = field(default=500)
    calib_sample_size: int = field(default=256)
    locat_sample_size: int = field(default=256)
    cache_cov_dir: str = field(default="./cache_cov")
    cache_projectors_dir: str = field(default="./cache_projectors")
    history_cov_datasets: List[str] = field(default_factory=lambda: ["mmlu"])
    cov_sample_sizes: List[int] = field(default_factory=lambda: [256])
    cov_seeds: List[int] = field(default_factory=lambda: [18])
