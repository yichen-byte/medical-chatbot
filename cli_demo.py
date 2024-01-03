from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError: 
    print("Install `readline` for a better experience.")


def main():
    chat_model = ChatModel()
    history = [('现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。', '当然，我将尽力为您提供关于中医的详细而清晰的回答。请问您有特定的中医问题或主题感兴趣吗？您可以提出您想了解的中医相关问题，比如中医理论、诊断方法、治疗技术、中药等方面的问题。我将根据您的需求提供相应的解答。')]
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

        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    main()
