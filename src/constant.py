########################### 对话SFT
INTENT_TOKEN = [
"<|Intent begin|>",
"<|Questioning|>",
"<|Admitting as being fact|>",
"<|Thinking/Saying the same|>",
"<|Consoling|>",
"<|Encouraging|>",
"<|Sympathizing|>",
"<|Wishing|>",
"<|Suggesting|>",
"<|Sharing own thoughts/opinion|>",
"<|Sharing or relating to own experience|>",
"<|Advising|>",
"<|Expressing care or concern|>",
"<|Expressing relief|>",
"<|Disapproving|>",
"<|Appreciating|>",
"<|Intent end|>",
]


CONV_SFT_SYSTEM_PROMPT = "You are an empathetic assistant."

SPECIAL_TOKEN = {
    "Llama-3.2-1B-Instruct": {"pad_token_id": 128004, "eos_token_id": 128009},
    "Qwen2.5-0.5B-Instruct": {"pad_token_id": 151643, "eos_token_id": 151645}
}


# response_token下一个token一定是非模板的内容，需要模型自己生成
CHAT_TOKENS = {
    "Llama-3.2-1B-Instruct": {"response_token": "<|start_header_id|>assistant<|end_header_id|>\n\n<|Intent begin|>", "human_token": "<|start_header_id|>user<|end_header_id|>\n\n"},
    "Qwen2.5-0.5B-Instruct": {"response_token": "<|im_start|>assistant\n<|Intent begin|>", "human_token": "<|im_start|>user\n"}
}

########################### 数据生成
LLM_INFO = {
    "deepseek-chat":{
        "base_url":"https://api.deepseek.com",
        "api_key":"sk-cadbbb9844d64124a62f284afda9cbba"
    }
}


########################### 奖励模型
COT_INSTRUCT = "Given the conversation between user and AI assistant, analyze whether AI assistant's response intention and response content are reasonable step by step.\n\nThe conversation between user and AI assistant is as follows\n"
COT_TRIGGER = "Let's analyze step by step.\n"
ANSWER_TRIGGER = "Final: Is the response reasonable (Yes/No)?" # 最后不能有空格
import re
ANALYSE_PATTERN = re.compile(
    r"Step (\d+):\n"
    r"Response intent: (.+)\n"
    r"Response content: (.+)\n"
    r"Analysis: .+\n"
    r"Reasonable: (Yes|No)\n"
)
########################### ln -s test