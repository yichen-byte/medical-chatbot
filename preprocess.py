import pandas as pd
import json

# 读取文件并按行解析JSON数据
data_list = []
with open('data/ChatMed_TCM-v0.2.json', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        data_list.append(data)

# 将列表转换为DataFrame
df = pd.DataFrame(data_list)
df['instruction'] = '现在你是一名专业的中医医生，请用你的专业知识提供详尽而清晰的关于中医问题的回答。'
df.columns = ['input', 'output', 'instruction']
# 调整列的顺序，将'instruction'列移到最前面
df = df[['instruction', 'input', 'output']]

# 输出DataFrame为JSON文件
df.to_json('data/medicalQA.json', orient='records', lines=True, force_ascii=False) 
