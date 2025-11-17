##写一个代码，下载deepseek-r1分词器，并测试聊天格式注入
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True, use_fast=False)

messages = [
    {"role": "user", "content": "你好，介绍一下你自己"},
    {"role": "assistant", "content": "你好，我是deepseek-r1，一个由deepseek公司开发的大语言模型。"}
]

tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(tokens)