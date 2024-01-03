import os
import platform
from transformers import AutoTokenizer, AutoModel
import torch
from fastllm_pytools import llm
import readline
import requests  
from lxml import etree  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.environ.get('MODEL_PATH', 'medical-chatbot')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 寻找以 .flm 结尾的文件
model_flm = [f for f in os.listdir(MODEL_PATH) if f.endswith(".flm")]
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
if len(model_flm) != 0:
    model = llm.model("model.flm") # 导入fastllm模型
else:
    if 'cuda' in DEVICE:
        # 是否对模型进行4-bit量化 
        # model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).quantize(4)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
    else: # CPU, Intel GPU and other GPU can use Float16 Precision Only
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE)
    model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"
    # model.save(os.path.join(MODEL_PATH, "model.flm")) # 导出fastllm模型


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。"


def build_prompt(query):
    try:
        # 构建查询URL  
        url = f"https://www.baidu.com/sf?openapi=1&dspName=iphone&from_sf=1&pd=wenda_kg&resource_id=5243&word={query}&dsp=iphone&title={query}&aptstamp=1704029696&top=%7B%22sfhs%22%3A11%7D&alr=1&fromSite=pc&total_res_num=5011&ms=1&frsrcid=5242&frorder=5&lid=10669184107456470945&pcEqid=94108e6b002f9fa10000000365916e00"  
        # 发送GET请求并获取响应内容  
        response = requests.get(url)  
        content = response.text  
        # 使用lxml解析HTML  
        html = etree.HTML(content)  
        # 提取答案  
        answers = html.xpath("/html/body/div[2]/div/div/b-superframe-body/div/div[2]/div/div/article/section/section/div/div/a/div[2]")  
        answer_texts = {f"相似回答{i+1}": answer.text for i, answer in enumerate(answers[:5])}
    except Exception as e:
        answer_texts = "无相似回答"

    # 构建prompt
    prompt = f'现在你是一名专业的中医医生，请回答以下患者的问诊问题：“{query}"，这里有一些相似的回答可能会帮助到你，注意请以你的知识为主，相似回答仅作为参考。相似回答：'
    print(answer_texts)
    return prompt

def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n患者：")
        if query.strip() == "exit":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\n医师：", end="")
        query = build_prompt(query)
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(str(response[current_length:]), end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()
