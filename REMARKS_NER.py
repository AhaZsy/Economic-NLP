import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import torch

# 读取Excel文件
file_path = "/home/zsy/NER/Updated_Results_1.xlsx"
df = pd.read_excel(file_path, sheet_name="REMARKS")
# 选择索引为39和40的行
#df = df.iloc[39:41]

# 定义NER模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")

# 分段和句子编号函数
def process_text(text):
    paragraphs = text.split("\n")
    para_sentences = []
    para_sentence_idx = 1
    for paragraphs_idx, para in enumerate(paragraphs):
        para_sentence_idx = 1
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
        for sentence in sentences:
            if sentence.strip():
                para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip(), None, None))  # 添加两个占位符，使其具有5个值
                para_sentence_idx += 1
    return para_sentences

# 对每一行的正文列进行处理
# processed_texts = []
# for idx, row in df.iterrows():
#     text = row['正文']
#     title = row['标题']
#     para_sentences = process_text(text)
#     processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])
# 对每一行的正文列进行处理
processed_texts = []
for idx, row in df.iterrows():
    text = str(row['正文'])  # 将文本转换为字符串
    title = row['标题']
    para_sentences = process_text(text)
    processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])


# 对处理后的文本进行NER识别
results = []
for title, idx, para_idx, sentence_idx, sentence in processed_texts:
    org_phrases = []
    misc_phrases = []
    org_temp = []
    misc_temp = []
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)
        labels = torch.argmax(outputs.logits, dim=2)
        
        # 迭代每个标记和预测，识别实体
        for token, prediction in zip(tokens, labels[0].cpu().numpy()):
        # 打印tokens和对应的标签
            print("Tokens:", tokens)
            if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
                if org_temp and '##' in token:
                    org_temp[-1] = org_temp[-1] + token[2:]
                else:
                    org_temp.append(token)
            else:
                if org_temp:
                    org_phrases.append(" ".join(org_temp))
                    org_temp = []

            if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
                if misc_temp and '##' in token:
                    misc_temp[-1] = misc_temp[-1] + token[2:]
                else:
                    misc_temp.append(token)
            else:
                if misc_temp:
                    misc_phrases.append(" ".join(misc_temp))
                    misc_temp = []

    # 将识别结果添加到结果列表中
    results.append((title, idx, para_idx, sentence_idx, sentence, labels, org_phrases, misc_phrases))

# 将结果输出到 CSV 文件之前的逻辑
result_df = pd.DataFrame(results, columns=['title', 'Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])

# 打印NER结果
# for result in results:
#     print("NER Result:", result)

# 检查并修正输出的行数
for idx, row in result_df.iterrows():
    sentence_length = len(row['Sentence'].split())
    org_length = sum(len(org.split()) for org in row['ORG'])
    misc_length = sum(len(misc.split()) for misc in row['MISC'])

    # 如果输出的长度超过了句子长度，将输出放置在下一行
    if org_length > sentence_length or misc_length > sentence_length:
        result_df.at[idx, 'Sentence'] = ''
        if len(result_df) <= idx + 1:
            result_df = result_df.append(pd.Series(), ignore_index=True)
        result_df.at[idx + 1, 'Sentence'] = row['Sentence']

# 将NER_Labels转换为字符串
result_df['NER_Labels'] = result_df['NER_Labels'].apply(lambda x: ', '.join(map(str, x)))

# 将结果保存到新的 CSV 文件
result_df.to_csv("REMARKS_NER_Final.csv", index=False, encoding='utf-8-sig')
