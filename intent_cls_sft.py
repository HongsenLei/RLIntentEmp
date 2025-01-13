import logging
import torch
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer, EvalPrediction
from src.constant import INTENT_TOKEN,SPECIAL_TOKEN
from dataclasses import dataclass, field
from typing import Optional
from src.utils import set_seed
from datasets import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification, LlamaConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# import debugpy
# try:
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

logger = logging.getLogger(__name__)

LABLE2ID = {label: i for i, label in enumerate(INTENT_TOKEN[1:-1])}
ID2LABLE = {i: label for i, label in enumerate(INTENT_TOKEN[1:-1])}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/d/hf_model/Llama-3.2-1B-Instruct")
    base_model: Optional[str] = field(default="Llama-3.2-1B-Instruct")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={
        "help": "The path to the training data"})
    valid_data_path: str = field(default=None, metadata={
        "help": "The path to the validation data"})
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    num_workers: int = field(default=6)

@dataclass
class TrainDataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features):
        # 从每个样本中提取字段
        input_ids = [torch.tensor(f['input_ids']) for f in features]
        attention_mask = [torch.tensor(f['attention_mask']) for f in features]
        labels = [f['labels'] for f in features]
        labels = torch.LongTensor(labels)

        # 对 input_ids 和 attention_mask 进行填充
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id,padding_side="right")
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0,padding_side="right")

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels,
        }

def get_input_id(conversation,situation,emotion,content,tokenizer:AutoTokenizer)->str:
    text = "Given a conversation between a user and an AI assistant, categorize the intention of the specified AI assistant reply."
    for turn in conversation:
        text += f"{turn['role']}: {turn['content']}\n"

    text += "Here are the situations and emotions the user wants to express, without the AI assistant knowing in advance.\n"
    text += f"What user wants to say is that '{situation}'.\n"
    text += f"The initial emotion of user is related to '{emotion}', user's emotion may change as the conversation progresses.\n\n" # 有关于user情感的标签质量非常低，不使用

    text += f"""What was the intent of the sentence "{content}" in the AI assistant's last reply?"""

    input_id = tokenizer.encode(text, add_special_tokens=True) # 只在最前面加<|begin_of_text|>, 不在最结尾加结束标志eos_token, 充分利用LLM生成能力

    return input_id

def _train_tokenize_fn(batch, model_max_length:int, tokenizer:AutoTokenizer):
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    # * 解包， zip打包， item_values是一个样本的所有值
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        conversation = item['conversation'] 
        input_id = get_input_id(conversation,item['situation'],item['emotion'],item['content'],tokenizer)
        while len(input_id)>model_max_length and len(conversation)>1:
            conversation = conversation[1:]
            input_id = get_input_id(conversation,item['situation'],item['emotion'],item['content'],tokenizer)
        
        attention_mask = [1]*len(input_id)
        label = LABLE2ID[f"<|{item['intent']}|>"]

        new_batch['input_ids'].append(input_id)
        new_batch['attention_mask'].append(attention_mask)
        new_batch['labels'].append(label)

    return new_batch

def get_dataset(data_args:DataArguments, training_args:TrainingArguments, model_args: ModelArguments, tokenizer:AutoTokenizer)->Dataset:
    set_seed(training_args.seed)
    train_dataset = Dataset.from_json(data_args.train_data_path)
    valid_dataset = Dataset.from_json(data_args.valid_data_path)
    tokenized_train_dataset = train_dataset.map(
            _train_tokenize_fn, 
            fn_kwargs={'model_max_length': data_args.model_max_length,'tokenizer': tokenizer}, 
            batched=True, 
            remove_columns=train_dataset.column_names, 
            num_proc=data_args.num_workers, 
            load_from_cache_file=False
        )
    tokenized_valid_dataset = valid_dataset.map(
            _train_tokenize_fn, 
            fn_kwargs={'model_max_length': data_args.model_max_length,'tokenizer': tokenizer}, 
            batched=True, 
            remove_columns=valid_dataset.column_names, 
            num_proc=data_args.num_workers, 
            load_from_cache_file=False
        )
    return tokenized_train_dataset, tokenized_valid_dataset


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,use_fast=True)
    with training_args.main_process_first():   
        tokenizer.pad_token_id = SPECIAL_TOKEN[model_args.base_model]["pad_token_id"]
        tokenizer.eos_token_id = SPECIAL_TOKEN[model_args.base_model]["eos_token_id"]
    

    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = len(ID2LABLE)  
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = LlamaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True, 
        device_map = "auto",
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32
    )
    
    with training_args.main_process_first(desc="loading data and tokenize"):
        train_dataset, valid_dataset = get_dataset(data_args, training_args, model_args,tokenizer)   

    data_collator = TrainDataCollator(tokenizer=tokenizer)

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=valid_dataset,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics
                    )
    trainer.train()
    # trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()