import torch
import torch.nn.functional as F
import logging
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import Tuple, Callable
import os
import random
import numpy as np
import json
import string
from datetime import datetime

accelerator: Accelerator = None

def setup_accelerator():
    global accelerator
    
    # hostname = socket.gethostname()
    # ip_address = socket.gethostbyname(hostname)
    ip_address = "localhost"
    # rank = int(os.environ['SLURM_PROCID'])
    # local_rank = int(os.environ['SLURM_LOCALID'])
    # world_size = int(os.environ['SLURM_NTASKS'])
    rank = int(os.environ.get('RANK', -1))  # 默认值为 -1 如果没有设置
    world_size = int(os.environ.get('WORLD_SIZE', -1))  # 默认值为 -1 如果没有设置
    local_rank = int(os.environ.get('LOCAL_RANK', -1))  
    port=29500
    host_addr_full = 'tcp://' + ip_address + ':' + str(port)
    
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{local_rank}"))                  
    assert torch.distributed.is_initialized()

    if accelerator is None:
        accelerator = Accelerator(split_batches=True)
    
    return accelerator

def synchronize_if_distributed():
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()
        
def synchronize_forward_on_stage3(done: bool, fake_forward_fn: Callable, **kwargs):
    # synchronize to avoid deadlock on deepspeed stage3. do not call this if zero-3 is disabled
    # https://github.com/microsoft/DeepSpeed/issues/860
    if done:
        sync = 1.
        while sync > 1e-5:
            fake_forward_fn(**kwargs)
            sync = torch.tensor(0., device=accelerator.device)
            sync = accelerator.reduce(sync).item()
    else:
        sync = torch.tensor(1., device=accelerator.device)
        sync = accelerator.reduce(sync)

def to_cuda(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(accelerator.device, non_blocking=True)

histroy_logs = set()
def print_rank_0(info, only_on_cuda0=False):
    if accelerator and not accelerator.is_main_process:
        return
    if only_on_cuda0 and info not in histroy_logs:
        histroy_logs.add(info)
        logging.info(info)
    return

def get_eval_ds_config(offload=None, stage=3):
    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }

@torch.no_grad()
def get_global_statistics(accelerator, xs: torch.Tensor, mask=None, device='cpu') -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    https://github.com/microsoft/LMOps/blob/cde1fb1ef4608a7ac5bf00675fa3e94b1d960abb/minillm/minillm/utils.py#L108
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count
    
    return global_mean.to(device), global_var.to(device), count.to(device)

class RunningMoments:
    def __init__(self, accelerator):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.accelerator = accelerator

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

@torch.no_grad()
def whiten(xs: torch.Tensor, mask: torch.BoolTensor, shift_mean=True, accelerator=None) -> torch.Tensor:
    """
    Whitens values
    """
    if accelerator != None and accelerator.use_distributed:
        mean, var, _ = get_global_statistics(accelerator, xs, mask=mask, device=accelerator.device)
    else:
        mean = xs.sum() / mask.sum()
        var = torch.sum(((xs - mean) ** 2).mul(mask)) / mask.sum()

    whitened = (xs - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened

def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
        
    return cum_logits

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=-1)
    logpy = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy

def get_category_distribution_entropy(bsz, logits):
    """
    Compute category distribution entropy
    """
    logits_distribution = torch.distributions.categorical.Categorical(logits=logits.reshape(-1, logits.size(-1)))
    ent = logits_distribution.entropy().reshape(bsz, -1)
    return ent

def pad_sequences(seqs, pad_value, padding='right', pad_to: int=None):
    """
    Padding sequence to the same length
    """
    max_len = max(len(seq) for seq in seqs) if pad_to is None else pad_to
    if padding == 'right':
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    elif padding == 'left':
        padded_seqs = [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]
    else:
        assert ValueError
    return padded_seqs

def prepare_forward(inputs,pad_token_id):
    # 左中右padding均可用
    attention_mask = inputs.ne(pad_token_id)
    # # trl ppo 方式[0,   0,   0,   1,   2,   3]
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    # batch generate 方式 [1,   1,   0,   1,   2,   3]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return attention_mask, position_ids

# KMP
def compute_prefix_function(pattern):
    m = len(pattern)
    next_arr = [0] * m  # 初始化next数组，表示每个字符前缀的最大长度
    k = 0  # 记录当前的前缀长度

    for q in range(1, m):
        # 处理失配，尝试回退
        while k > 0 and pattern[k] != pattern[q]:
            k = next_arr[k - 1]  # 回退到前一个最长前缀的位置
        
        # 匹配成功，增加前缀长度
        if pattern[k] == pattern[q]:
            k += 1
        
        next_arr[q] = k  # 记录当前字符的前缀长度
    
    return next_arr

def KMP_matcher(text, pattern):
    n = len(text)
    m = len(pattern)
    next_arr = compute_prefix_function(pattern)
    q = 0  # 已匹配的字符个数

    match_positions = []  # 用于存储匹配的起始位置

    for i in range(n):
        # 当出现失配时，按照next数组回退
        while q > 0 and pattern[q] != text[i]:
            q = next_arr[q - 1]
        
        # 成功匹配一个字符，继续匹配下一个
        if pattern[q] == text[i]:
            q += 1
        
        # 如果成功匹配整个模式
        if q == m:
            match_positions.append(i - m + 1)  # 记录匹配的起始位置
            q = next_arr[q - 1]  # 回退到最长前缀的位置，继续匹配
    
    return match_positions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)

def read_jsonl2list(file: str) -> list:
    data = []
    with open(file, 'r', encoding='utf-8') as read_file:
        for line in read_file:
            data.append(json.loads(line))
    return data

def write_jsonl_append_line(file: str, data: dict) -> None:
    with open(file, "a", encoding="utf-8") as write_file:
        write_file.write(json.dumps(data, ensure_ascii=False) + '\n')


def generate_uid() -> str:
    # 获取当前日期和时间（格式：YYYYMMDD_HHMMSS）
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 生成一个乱序的8位字母
    letters = random.choices(string.ascii_letters, k=8)
    random_letters = ''.join(letters)
    
    # 拼接日期时间和乱序字母，生成UID
    uid = f"{now}{random_letters}"
    return uid

if __name__=="__main__":
    text = [5,6,8,9,3,5,9,42,6,3,5,9,42,6]
    pattern = [3,5,9,42,6]
    begin_idx = KMP_matcher(text,pattern)
    print(begin_idx)
    end_idx= []
    for i in begin_idx:
        end_idx.append(i+len(pattern))
    print(end_idx)
    print(text[end_idx])