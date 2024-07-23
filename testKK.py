import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import torch

# # 读取Excel文件
file_path = "/home/yk/llm_code/YaoYao/test.xlsx"
df = pd.read_excel(file_path, sheet_name="P-A")

# # 定义NER模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/home/yk/llm_code/YaoYao/electra-large-discriminator-finetuned-conll03-english")

model = AutoModelForTokenClassification.from_pretrained("/home/yk/llm_code/YaoYao/electra-large-discriminator-finetuned-conll03-english")

# 分段和句子编号函数
def process_text(text):
    # 按段落分割
    paragraphs = text.split("\n")
    print(len(paragraphs))
    para_sentences = []
    para_sentence_idx = 1
    for paragraphs_idx, para in enumerate(paragraphs):
        # 按句子分割
        para_sentence_idx = 1
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
        for sentence in sentences:
            if sentence.strip():
                para_sentences.append((paragraphs_idx,para_sentence_idx, sentence.strip()))
                para_sentence_idx += 1
    return para_sentences
# 对每一行的正文列进行处理
processed_texts = []
for idx, row in df.iterrows():
    text = row['正文']
    title = row['标题']
    para_sentences = process_text(text)
    processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence in para_sentences])
# # 对处理后的文本进行NER识别
results = []
for title, idx, para_idx, sentence_idx, sentence in processed_texts:
    org = []
    misc = []
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    tokens = inputs.tokens()
    with torch.no_grad():
        outputs = model(**inputs)
        labels = torch.argmax(outputs.logits, dim=2)
    
# 添加新的两个列，一个是ORG一个是MISC， org以及misc获取的办法
    
        for token, prediction, index in zip(tokens, labels[0].cpu().numpy(),range(len(tokens))):
            if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
                if prediction == labels[0].cpu().numpy()[index-1] and index != 0:
                    if '##' in token:
                        org[-1] = org[-1] + token[2:]
                    else:
                        org[-1] = org[-1] + " " + token
                else:
                    org.append(token)
            if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
                if prediction == labels[0].cpu().numpy()[index-1] and index != 0:
                    if '##' in token:
                        misc[-1] = misc[-1] + token[2:]
                    else:
                        misc[-1] = misc[-1] + " " + token
                else:
                    misc.append(token)
    results.append((title, idx, para_idx, sentence_idx, sentence, labels, org, misc))
    print(results)
# # 将结果保存到新的CSV文件
result_df = pd.DataFrame(results, columns=['Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])
result_df.to_csv("result_ARTICLES_NER.csv", index=False)

