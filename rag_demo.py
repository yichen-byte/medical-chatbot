from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
import requests
from lxml import etree 

try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")


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
        answers = html.xpath("/html/body/div[2]/div/div/b-superframe-body/div/div[2]/div/div/article/section/section/div/div/a/div[2]/text()")[:3]  
        if len(answers) == 0:
            answer_texts = "无相似回答"
        else:
            answer_texts = {f"相似回答{i+1}": answer for i, answer in enumerate(answers)}
    except Exception as e:
        print(e)
        answer_texts = "无相似回答"

    # 构建prompt
    prompt = f'现在你是一名专业的中医医生，请回答以下患者的问诊问题：“{query}"。这里有一些相似的回答可能会帮助到你，需要注意的是，在你提供的答案中，请以你的中医知识为主，相似回答仅作为参考。相似回答：{answer_texts}'
    # prompt = f'现在你是一名专业的中医医生，请回答以下患者的问诊问题：“{query}"。这里有一些相似的回答可能会帮助到你：{answer_texts}'
    return prompt

def main():
    chat_model = ChatModel()
    history = []
    print("欢迎使用中医聊天机器人，使用 clear 命令可清除聊天历史，使用 exit 命令可退出应用程序。")

    while True:
        try:
            query = input("\n患者: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            torch_gc()
            print("History has been removed.")
            continue

        print("医师: ", end="", flush=True)
        query = build_prompt(query)
        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    main()
