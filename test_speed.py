import os
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from fastllm_pytools import llm

try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")

MODEL_PATH = os.environ.get('MODEL_PATH', './medical-chatbot')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH,trust_remote_code=True).quantize(4)


# model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。"
# 从文件中读取queries
with open('./test.txt', 'r') as file:
    querys = file.read().split('\n')

def main():
    past_key_values, history = None, []
    history = [('现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。', '当然，我将尽力为您提供关于中医的详细而清晰的回答。请问您有特定的中医问题或主题感兴趣吗？您可以提出您想了解的中医相关问题，比如中医理论、诊断方法、治疗技术、中药等方面的问题。我将根据您的需求提供相应的解答。')]

    global stop_stream
    total_token_count = 0
    total_elapsed_time = 0

    for query in querys:
        start_time = time.time()  # 记录开始时间
        token_count = 0
        for response, _, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,  
                                                                    temperature=0.01,  
                                                                    past_key_values=past_key_values,  
                                                                    return_past_key_values=True):  
            if stop_stream:  
                stop_stream = False  
                break  
            else:
                token_count += 1  # 计算token数量
        
        elapsed_time = time.time() - start_time  # 计算总耗时
        total_elapsed_time += elapsed_time
        total_token_count += token_count

    # 计算平均速度
    average_speed = total_token_count / total_elapsed_time if total_elapsed_time > 0 else 0  # token/s
    print(f"Average Speed: {average_speed:.2f} token/s")  # 打印平均速度

if __name__ == "__main__":
    main()
