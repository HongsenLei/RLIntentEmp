import logging
import torch
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from sft.emp_sft_args import DataArguments, ModelArguments
from sft.emp_sft_data import get_sft_train_dataset, SFTTrainDataCollator
from src.constant import INTENT_TOKEN,SPECIAL_TOKEN


# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

logger = logging.getLogger(__name__)


def prepare_model(model_args: ModelArguments, training_args: TrainingArguments, tokenizer: AutoTokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True, 
        device_map = "auto",
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32
    )
    with training_args.main_process_first():
        if model_args.add_special_token:
            # 3. 调整模型的嵌入层大小，以适应新添加的 Token
            model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    
    return model

def prepare_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,use_fast=True)

    with training_args.main_process_first():   
        tokenizer.pad_token_id = SPECIAL_TOKEN[model_args.base_model]["pad_token_id"]
        tokenizer.eos_token_id = SPECIAL_TOKEN[model_args.base_model]["eos_token_id"]
        if model_args.add_special_token:
            # 2. 添加新 Token
            num_added_tokens = tokenizer.add_tokens(INTENT_TOKEN)
            print("We have added", num_added_tokens, "tokens")
            print(f'[Vocab size]: {len(tokenizer)}')   
    return tokenizer

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = prepare_tokenizer(model_args, training_args)
    with training_args.main_process_first(desc="loading data and tokenize"):
        train_dataset = get_sft_train_dataset(data_args,training_args, model_args,tokenizer)

    data_collator = SFTTrainDataCollator(tokenizer=tokenizer)
    model = prepare_model(model_args, training_args, tokenizer)

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    # trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()