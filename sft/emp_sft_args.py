from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/d/hf_model/Llama-3.2-1B-Instruct")
    base_model: Optional[str] = field(default="Llama-3.2-1B-Instruct")
    add_special_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "True if first train with origin llm"
        }
    )

@dataclass
class DataArguments:
    conv_train_data_path: str = field(default=None, metadata={
        "help": "Path to the training data of conversation."})
    rm_cot_train_data_path: str = field(default=None, metadata={
        "help": "Path to the training data of gen reward model."})
    data_mode: str = field(default="conv")
    rm_ratio: float = field(default=0.0, metadata={"help": "The ratio for gen reward model data."})
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    num_workers: int = field(default=6)

    def __post_init__(self):
        # This is where we add our range check
        if not (0 <= self.rm_ratio <= 1):
            raise ValueError(f"rm_ratio must be between 0 and 1, got {self.rm_ratio}")
        if self.data_mode not in ["conv","rm","mix"]:
            raise ValueError(f"""data_mode must be in ("conv","rm","mix"), got {self.data_mode}""")
