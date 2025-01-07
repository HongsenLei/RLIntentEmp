import openai
from .constant import LLM_INFO
import json

class ChatClient:
    def __init__(self, model_name, system_prompt, temperature=1.0, max_tokens=1000, top_p=1.0):
        """
        初始化 ChatClient 对象，设置 OpenAI 客户端和系统提示
        
        :param model_name: 模型名字
        :param system_prompt: 系统prompt，指导模型的行为
        :param temperature: 控制生成内容的随机性，默认值为 1.0
        :param max_tokens: 限制生成内容的最大令牌数，默认值为 1000
        :param top_p: 用于控制输出的多样性，默认值为 1.0
        """
        # 设置 OpenAI API 密钥和基础 URL
        self.client = openai.OpenAI(api_key=LLM_INFO[model_name]["api_key"], base_url=LLM_INFO[model_name]["base_url"])
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def get_response(self, text):
        """
        根据用户输入的文本获取模型回复
        
        :param text: 用户的输入文本
        :return: 模型的回复内容或 None 如果出错
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # 使用适当的模型名称
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            # 捕获异常并返回 None
            print(f"An error occurred: {e}")
            return None

def convert_llmjson2dict(llm_response):
   # 找到第一个 '{' 和最后一个 '}'
    start = llm_response.find('{')
    end = llm_response.rfind('}')
    
    # 如果找到了 '{' 和 '}'，则截取子字符串
    if start != -1 and end != -1 and start < end:
        llm_response = llm_response[start:end+1]
        try:
            llm_response_dict = json.loads(llm_response)
            return llm_response_dict
        except Exception as e:
            # print("字典转换失败")
            return None
    # print("json提取失败")
    return None
