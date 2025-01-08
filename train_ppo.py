import random
import logging
import numpy as np
import torch
import torch.nn as nn
from config_ppo import parse_args
from ppo.ppo_trainer import PPOTrainer
from ppo.ppo_datahelper import get_tokenizer
from src.utils import setup_accelerator,synchronize_if_distributed,print_rank_0,prepare_forward
from accelerate.state import AcceleratorState
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification
from transformers import AutoModelForCausalLM
import os


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# class LlamaValueModel(LlamaForCausalLM):
#     def __init__(self, config, opt, tokenizer):
#         super().__init__(config)
#         self.opt = opt
#         self.tokenizer = tokenizer
#         self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        
#     def forward(self, decoder_input, only_last=True):
#         attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
#         output = self.model.forward(
#             input_ids=decoder_input,
#             attention_mask=attention_mask, 
#             return_dict=True,
#             use_cache=False
#             )
        
#         if only_last:
#             logits = self.value_head(output.last_hidden_state[:, -1, :]).squeeze(-1)
#         else:
#             logits = self.value_head(output.last_hidden_state).squeeze(-1)
        
#         return (logits,)

class LlamaValueModel(LlamaForSequenceClassification):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        # self.score = nn.Linear(config.hidden_size, 1, bias=False)
        
    def forward(self, decoder_input, only_last=True):
        attention_mask, position_ids = prepare_forward(decoder_input,self.tokenizer.pad_token_id)
        output = self.model.forward(
            input_ids=decoder_input,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            return_dict=True,
            use_cache=False
            )
        
        if only_last:
            logits = self.score(output.last_hidden_state[:, -1, :]).squeeze(-1)
        else:
            logits = self.score(output.last_hidden_state).squeeze(-1)
        
        return (logits,)
    
def main(opt):
    # setup accelerator
    accelerator = setup_accelerator()

    # setup deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = opt.batch_size
    deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}

    # logging config
    logging.basicConfig(
            format='%(asctime)s - ' + f'Rank: {accelerator.process_index}' + ' - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO
            )
    logger = logging.getLogger(__name__)

    # fix seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # tokenizer
    tokenizer = get_tokenizer(opt)

    # load policy model
    logging.info(f"Loading policy model from: {opt.policy_model_path}...")
    policy_model = AutoModelForCausalLM.from_pretrained(opt.policy_model_path)
    policy_model._set_gradient_checkpointing(opt.gradient_checkpoint)

    # load critic model
    logging.info(f"Loading critic model from: {opt.critic_model_path}...")
    critic_model = LlamaValueModel.from_pretrained(opt.critic_model_path, opt, tokenizer, torch_dtype=torch.bfloat16)
    critic_model._set_gradient_checkpointing(opt.gradient_checkpoint)

    # load reference model
    logging.info(f"Loading reference model from: {opt.policy_model_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(opt.policy_model_path)

    synchronize_if_distributed()
    trainer = PPOTrainer(opt, policy_model, ref_model, critic_model, accelerator)
    trainer.train()

    logging.info('==================Congrats! Training completed, exit process...==================') 

if __name__ == '__main__':
    opt = parse_args()
    print_rank_0(opt)
    main(opt)