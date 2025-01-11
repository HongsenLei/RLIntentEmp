from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvaluatingArgument:
    output_dir: str
    batch_size: int = field(default=8)
    seed: int =field(default=42)
    max_new_token: int =field(default=1024)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/d/hf_model/Llama-3.2-1B-Instruct")
    base_model: Optional[str] = field(default="Llama-3.2-1B-Instruct")

@dataclass
class DataArguments:
    conv_eval_data_path: str = field(default=None, metadata={
        "help": "Path to the evaluating data of conversation."})
    rm_cot_eval_data_path: str = field(default=None, metadata={
        "help": "Path to the evaluating data of gen reward model."})
    data_mode: str = field(default="conv")
    conv_sample_mode: str = field(default="valid")
    eval_max_length:int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be left truncated."},
    )
    num_workers: int = field(default=6)

    def __post_init__(self):
        # This is where we add our range check
        if self.data_mode not in ["conv","rm","mix"]:
            raise ValueError(f"""data_mode must be in ("conv","rm","mix"), got {self.data_mode}""")