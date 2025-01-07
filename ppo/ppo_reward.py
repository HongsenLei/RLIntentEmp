from typing import List, Dict
# from constant import ANSWER_TRIGGER, COT_INSTRUCT, ANSWER_TRIGGER
# from utils import KMP_matcher
import re
import torch
from transformers import GenerationConfig
import math
pattern = re.compile(
    r"Step (\d+):\n"
    r"Response intent: (.+)\n"
    r"Response content: (.+)\n"
    r"Analysis: .+\n"
    r"Reasonable: (Yes|No)\n"
)

def reward_map_infinite(x:float)->float:
    """
    将 [0, 1] 区间的数据映射到 (-∞, +∞)。

    参数:
    x (float): 输入值，必须在 (0, 1) 范围内。
    
    返回:
    float: 映射到 (-∞, +∞) 的值。
    """
    # 设置一个小的 epsilon，避免数值溢出
    epsilon = 1e-9
    x = max(epsilon, min(x, 1 - epsilon))
    return 12*math.log(x / (1 - x))-12

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

COT_INSTRUCT = "Given the conversation between user and AI assistant, analyze whether AI assistant's response intention and response content are reasonable step by step.\n\nThe conversation between user and AI assistant is as follows\n"
COT_TRIGGER = "Let's analyze step by step.\n"
ANSWER_TRIGGER = "Final: Is the response reasonable (Yes/No)?" # 最后不能有空格

def get_rule_reward(response2Banalyzed:List[str], sampled_text:List[str], label:List[str], eos_token:str)->List[float]:
    reward = []
    for _response, _sample, _label in zip(response2Banalyzed, sampled_text, label):
        response_intent = re.findall(r"<\|([^|]+)\|>", _response)
        response_intent = response_intent[1:-1] # 去掉<|Intent begin|> <|Intent end|>
        cleaned_message = [part.strip() for part in re.split(r'<\|.*?\|>', _response) if part.strip()]
        assert len(response_intent)==len(cleaned_message)
        num_step = len(response_intent)
        _reward = 0.0
        if _sample.endswith(eos_token):
            _reward += 0.1 # 有终止符
            pred_label = _sample.split(ANSWER_TRIGGER)[-1].strip().replace(eos_token,"")
            if _label==pred_label: # 最终预测正确
                _reward += 0.4
                matches = pattern.findall(_sample)
                if len(matches)==num_step:
                    _reward += 0.1 # 步骤分析正确, 剩余分析奖励最高0.4
                    for stp, match in enumerate(matches):
                        m_step, m_intent, m_content, _ = match # TODO
                        if int(m_step)-1==stp:
                            _reward += (0.4/num_step)*0.1
                        if m_intent.strip() == response_intent[stp].strip():
                            _reward += (0.4/num_step)*0.6
                        if m_content.strip() == cleaned_message[stp].strip():
                            _reward += (0.4/num_step)*0.3
        reward.append(_reward)
    return reward


@torch.no_grad()
def get_generative_reward(
    generative_reward_model: torch.nn.Module,
    tokenizer, 
    generation_config: GenerationConfig, 
    sampled_text:List[str], 
    conversation_history:List[List[Dict]],
    )->List[float]:
    """
    sampled_text: 从第一个<|Intend|>开始的response
    <|Sympathizing|>I'm sure she will. <|Encouraging|>Keep supporting one another; it makes the community stronger.<|Intent end|><|eot_id|>
    
    conversation_history:
    [
    {'role': 'system', 'content': 'You are an empathetic assistant.'}, 
    {'role': 'user', 'content': 'My neighbor was in a difficult situation and needed money for food, so I decided to lend her some.'}, 
    {'role': 'assistant', 'content': "<|Intent begin|><|Appreciating|>That was very kind of you. <|Sharing own thoughts/opinion|>It's reassuring to know that there are still good neighbors around.<|Intent end|>", 'intent_list': ['Appreciating', 'Sharing own thoughts/opinion', 'Intent end']}, 
    {'role': 'user', 'content': 'Yes, we do need to support each other. I trust her and believe she will repay me.'}, 
    {'role': 'assistant', 'content': "<|Intent begin|><|Thinking/Saying the same|>I hope so too. <|Sharing own thoughts/opinion|>It's important to have that trust. <|Appreciating|>Your gesture is truly commendable.<|Intent end|>", 'intent_list': ['Thinking/Saying the same', 'Sharing own thoughts/opinion', 'Appreciating', 'Intent end']}, 
    {'role': 'user', 'content': 'Thank you. I just hope she can manage until she pays me back.'}
    ]
    """
    ###############
    # 写成类函数之后，常量写进初始化函数里
    ###############
    answer_trigger_token_ids = tokenizer.encode(ANSWER_TRIGGER, add_special_tokens=False)
    YES_SCORE_IDX = tokenizer.encode("Yes", add_special_tokens=False)[0]
    ###############
    rewards=[]
    device:torch.device = generative_reward_model.device
    input_ids_list = []
    bsz = len(sampled_text)
    for _response, _conv_history in zip(sampled_text,conversation_history):
        _conv_history = _conv_history[1:] # 去除system prompt
        _response = "<|Intent begin|>" + _response # 完整回复
        text = COT_INSTRUCT
        total_conv = "" 
        for turn in _conv_history:
            total_conv += f"{turn['role']}: {turn['content']}\n"
        total_conv += f"assistant: {_response}\n"
        text += total_conv
        text += f"Analyze whether the AI assitant's last response is reasonable:\n{_response}\n"
        text += COT_TRIGGER
        input_ids=tokenizer.encode(text, add_special_tokens=True)# 只在最前面加<|begin_of_text|>, 不在最结尾加结束标志eos_token
        input_ids_list.append(input_ids)
    batch_input_ids =torch.tensor(pad_sequences(input_ids_list,pad_value=tokenizer.pad_token_id,padding="left"),
                        dtype=torch.long, device=device)
    batch_attention_mask = batch_input_ids!=tokenizer.pad_token_id
    output_COT = generative_reward_model.generate(
        batch_input_ids,
        attention_mask = batch_attention_mask,
        pad_token_id = tokenizer.pad_token_id,
        generation_config = generation_config
    )
    sequences = output_COT.sequences
    scores = output_COT.scores
    context_len = len(batch_input_ids[0])
    num_return_sequences = generation_config.num_return_sequences
    # scores: len(scores) 新生成token数; scores[0].shape [bsz*num_return_sequences, 词表大小]; sequences: [bsz*num_return_sequences, 总长度] 总长度=len(input_ids)+新生成token数
    for bsz_idx in range(bsz):
        valid_count = 0 + 1e-9
        singel_reward = 0
        for seq_idx in range(num_return_sequences):
            match_idx_list = KMP_matcher(sequences[bsz_idx*num_return_sequences+seq_idx].tolist(),answer_trigger_token_ids)
            if len(match_idx_list)>0:
                valid_count += 1
                # 找到Yes对应scores的下标
                yes_token_idx = match_idx_list[-1]+len(answer_trigger_token_ids)-context_len
                softmax_score = torch.nn.functional.softmax(scores[yes_token_idx][bsz_idx*num_return_sequences+seq_idx], dim=-1)
                singel_reward += softmax_score[YES_SCORE_IDX].item() # 转换为python标量
        rewards.append(singel_reward/valid_count)
    return rewards


if __name__=="__main__":
#     response = [
#         "<|Intent begin|><|Sharing own thoughts/opinion|>That sounds like a typical day on the road. <|Wishing|>I hope you didn't get too shaken up.<|Intent end|>",
#         "<|Intent begin|><|Sympathizing|>That sounds like a great opportunity, but I can understand why you might feel apprehensive. <|Questioning|>Have you done any research on the company or the job offer?<|Intent end|>"
#     ]
#     sampled_text=[
# """Step 1:
# Response intent: Sharing own thoughts/opinion
# Response content: That sounds like a typical day on the road.
# Analysis: The AI assistant shares its opinion about the situation being 'typical,' which might trivialize the user's experience. It does not acknowledge the severity of the incident or express empathy.
# Reasonable: No

# Step 2:
# Response intent: Wishing
# Response content: I hope you didn't get too shaken up.
# Analysis: While expressing a wish for the user's well-being is generally empathetic, in this context, it comes across as dismissive of the user's potential distress. It does not fully understand the gravity of the situation.
# Reasonable: No

# Final: Is the response reasonable (Yes/No)?No<|eot_id|>""",
# """Step 1:
# Response intent: Sharing own thoughts/opinion
# Response content: That sounds like a great opportunity, but I can understand why you might feel apprehensive.
# Analysis: The AI assistant acknowledges the positive aspect of the offer but also expresses empathy towards the user's potential apprehension, which is appropriate given the context of an uncertain situation.
# Reasonable: Yes

# Step 2:
# Response intent: Questioning
# Response content: Have you done any research on the company or the job offer?
# Analysis: This question is relevant to the conversation and shows interest in understanding the user's experience better. It aligns with the user's initial statement about receiving an offer and seeking reassurance.
# Reasonable: Yes

# Final: Is the response reasonable (Yes/No)?Yes<|eot_id|>"""
#     ]
#     label=["No","Yes"]
#     rewards = get_rule_reward(response,sampled_text,label,"<|eot_id|>")
#     print(rewards)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("/seu_share/home/wutianxing/220222120/IntentEMP/result/sft_rm/checkpoint-900").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("/seu_share/home/wutianxing/220222120/IntentEMP/result/sft_rm/checkpoint-900")
    # print(tokenizer("Final: Is the response reasonable (Yes/No)?",add_special_tokens=False)) # {'input_ids': [19918, 25, 2209, 279, 2077, 13579, 320, 9642, 14, 2822, 12106], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # print(tokenizer("Final: Is the response reasonable (Yes/No)? ",add_special_tokens=False)) # {'input_ids': [19918, 25, 2209, 279, 2077, 13579, 320, 9642, 14, 2822, 12106, 220], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # print(tokenizer("Final: Is the response reasonable (Yes/No)?Yes",add_special_tokens=False)) # {'input_ids': [19918, 25, 2209, 279, 2077, 13579, 320, 9642, 14, 2822, 12106, 9642], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # print(tokenizer("Final: Is the response reasonable (Yes/No)? Yes",add_special_tokens=False)) # {'input_ids': [19918, 25, 2209, 279, 2077, 13579, 320, 9642, 14, 2822, 12106, 7566], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    reward_cot_generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_k=20,
        num_return_sequences=8,
        output_scores=True,
        return_dict_in_generate=True,
        return_legacy_cache=True,
    )
    # scores 和 logits 都没有经过softmax，scores是logits经过temperature、top_k等处理的结果。score比logits更好，防止Yes和No的概率都很小
    sampled_text=[
        "<|Appreciating|>Well, at least you didn't waste any money on it either!<|Intent end|><|eot_id|>", 
        '<|Sympathizing|>I have faith in people too. <|Questioning|>What did your friend owe you?<|Intent end|><|eot_id|>'
        ]
    conversation_history=[
        [
            {'role': 'system', 'content': 'You are an empathetic assistant.'}, 
            {'role': 'user', 'content': 'I recently purchased a cake from the store, but unfortunately, it turned out to be quite disappointing in terms of taste.'}, 
            {'role': 'assistant', 'content': "<|Intent begin|><|Sharing own thoughts/opinion|>It's somewhat of a gamble when buying store-bought cakes. <|Questioning|>Did you consider returning it?<|Intent end|>", 'intent_list': ['Sharing own thoughts/opinion', 'Questioning', 'Intent end']}, 
            {'role': 'user', 'content': 'I did not return it. Instead, I decided to endure the unpleasant taste and finished eating it.'}, {'role': 'assistant', 'content': "<|Intent begin|><|Sharing own thoughts/opinion|>You must have a lot more bravery than I do. <|Sharing or relating to own experience|>I don't think I could force myself to eat it.<|Intent end|>", 'intent_list': ['Sharing own thoughts/opinion', 'Sharing or relating to own experience', 'Intent end']}, 
            {'role': 'user', 'content': 'It was indeed a challenging experience, but I wanted to avoid wasting food.'}
        ], 
        [
            {'role': 'system', 'content': 'You are an empathetic assistant.'}, 
            {'role': 'user', 'content': 'My neighbor was in a difficult situation and needed money for food, so I decided to lend her some.'}, 
            {'role': 'assistant', 'content': "<|Intent begin|><|Appreciating|>That was very kind of you. <|Sharing own thoughts/opinion|>It's reassuring to know that there are still good neighbors around.<|Intent end|>", 'intent_list': ['Appreciating', 'Sharing own thoughts/opinion', 'Intent end']}, 
            {'role': 'user', 'content': 'Yes, we do need to support each other. I trust her and believe she will repay me.'}
        ]
        ]
    reward = get_generative_reward(
        generative_reward_model=model,
        tokenizer=tokenizer,
        generation_config=reward_cot_generation_config,
        sampled_text=sampled_text,
        conversation_history=conversation_history
    )
    print(reward)
    