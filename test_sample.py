import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import torch

# 读取Excel文件
file_path = "/home/zsy/NER/Updated_Results_1.xlsx"
df = pd.read_excel(file_path, sheet_name="ARTICLES")
df = df.iloc[613:614]
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
processed_texts = []
for idx, row in df.iterrows():
    text = row['正文']
    title = row['标题']
    para_sentences = process_text(text)
    processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])

# 添加测试句子
processed_texts.append(("Test Title", 0, 0, 0, "New Hampshire Speaker Shawn Jasper’s Letter to Comm. Barthelmes & Sec. Gardner"))

# 对处理后的文本进行NER识别
results = []
for title, idx, para_idx, sentence_idx, sentence in processed_texts:
    org_phrases = []
    misc_phrases = []
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)
        labels = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

    # 初始化识别结果列表
    org_temp = []
    misc_temp = []
    current_label = None

    # 迭代每个标记和预测，识别实体
    for token, prediction in zip(tokens, labels):
        label = model.config.id2label[prediction]
        
        if label.startswith('B-'):
            current_label = label[2:]
            temp_list = org_temp if current_label == "ORG" else misc_temp
            temp_list.append(token)
        elif label.startswith('I-'):
            if current_label:
                temp_list[-1] += token[2:]
        else:
            if current_label == "ORG" and org_temp:
                org_phrases.append(" ".join(org_temp))
                org_temp = []
            elif current_label == "MISC" and misc_temp:
                misc_phrases.append(" ".join(misc_temp))
                misc_temp = []
            current_label = None

    # 将识别结果添加到结果列表中
    results.append((title, idx, para_idx, sentence_idx, sentence, labels, org_phrases, misc_phrases))

# 将结果输出到 CSV 文件之前的逻辑
result_df = pd.DataFrame(results, columns=['title', 'Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])

# 将NER_Labels转换为字符串
result_df['NER_Labels'] = result_df['NER_Labels'].apply(lambda x: ', '.join(map(str, x)))

# 将结果保存到新的 CSV 文件
result_df.to_csv("ARTICLES_NER_Final_2.csv", index=False, encoding='utf-8-sig')


# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import re
# import torch

# # 读取Excel文件
# file_path = "/home/zsy/NER/Updated_Results_1.xlsx"
# df = pd.read_excel(file_path, sheet_name="ARTICLES")
# # 选择索引为39和40的行
# df = df.iloc[39:41]

# # 定义NER模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")
# model = AutoModelForTokenClassification.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")

# # 分段和句子编号函数
# def process_text(text):
#     paragraphs = text.split("\n")
#     para_sentences = []
#     para_sentence_idx = 1
#     for paragraphs_idx, para in enumerate(paragraphs):
#         para_sentence_idx = 1
#         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
#         for sentence in sentences:
#             if sentence.strip():
#                 para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip(), None, None))  # 添加两个占位符，使其具有5个值
#                 para_sentence_idx += 1
#     return para_sentences

# # 对每一行的正文列进行处理
# processed_texts = []
# for idx, row in df.iterrows():
#     text = row['正文']
#     title = row['标题']
#     para_sentences = process_text(text)
#     processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])

# # 对处理后的文本进行NER识别
# results = []
# for title, idx, para_idx, sentence_idx, sentence in processed_texts:
#     org_phrases = []
#     misc_phrases = []
#     org_temp = []
#     misc_temp = []
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

#     with torch.no_grad():
#         outputs = model(**inputs)
#         labels = torch.argmax(outputs.logits, dim=2)
        
#         # 迭代每个标记和预测，识别实体
#         for token, prediction in zip(tokens, labels[0].cpu().numpy()):
#             if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
#                 if org_temp and '##' in token:
#                     org_temp[-1] = org_temp[-1] + token[2:]
#                 else:
#                     org_temp.append(token)
#             else:
#                 if org_temp:
#                     org_phrases.append(" ".join(org_temp))
#                     org_temp = []

#             if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
#                 if misc_temp and '##' in token:
#                     misc_temp[-1] = misc_temp[-1] + token[2:]
#                 else:
#                     misc_temp.append(token)
#             else:
#                 if misc_temp:
#                     misc_phrases.append(" ".join(misc_temp))
#                     misc_temp = []

#     # 将识别结果添加到结果列表中
#     results.append((title, idx, para_idx, sentence_idx, sentence, labels, org_phrases, misc_phrases))

# # 将结果输出到 CSV 文件之前的逻辑
# result_df = pd.DataFrame(results, columns=['title', 'Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])

# # 检查并修正输出的行数
# for idx, row in result_df.iterrows():
#     sentence_length = len(row['Sentence'].split())
#     org_length = sum(len(org.split()) for org in row['ORG'])
#     misc_length = sum(len(misc.split()) for misc in row['MISC'])

#     # 如果输出的长度超过了句子长度，将输出放置在下一行
#     if org_length > sentence_length or misc_length > sentence_length:
#         result_df.at[idx, 'Sentence'] = ''
#         if len(result_df) <= idx + 1:
#             result_df = result_df.append(pd.Series(), ignore_index=True)
#         result_df.at[idx + 1, 'Sentence'] = row['Sentence']

# # 将NER_Labels转换为字符串
# result_df['NER_Labels'] = result_df['NER_Labels'].apply(lambda x: ', '.join(map(str, x)))

# # 将结果保存到新的 CSV 文件
# result_df.to_csv("test_ARTICLES_NER_corrected.csv", index=False, encoding='utf-8-sig')




# # import pandas as pd
# # from transformers import AutoTokenizer, AutoModelForTokenClassification
# # import re
# # import torch

# # # 读取Excel文件
# # file_path = "NER/Updated_Results_1.xlsx"
# # df = pd.read_excel(file_path, sheet_name="P-A")

# # # 仅处理前20篇文章
# # df = df.head(5)

# # # 定义NER模型和分词器
# # tokenizer = AutoTokenizer.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")
# # model = AutoModelForTokenClassification.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")

# # # 分段和句子编号函数
# # def process_text(text):
# #     paragraphs = text.split("\n")
# #     para_sentences = []
# #     para_sentence_idx = 1
# #     for paragraphs_idx, para in enumerate(paragraphs):
# #         para_sentence_idx = 1
# #         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
# #         for sentence in sentences:
# #             if sentence.strip():
# #                 para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip()))
# #                 para_sentence_idx += 1
# #     return para_sentences

# # # 对每一行的正文列进行处理
# # processed_texts = []
# # for idx, row in df.iterrows():
# #     text = row['正文']
# #     title = row['标题']
# #     para_sentences = process_text(text)
# #     processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence in para_sentences])

# # # 对处理后的文本进行NER识别
# # results = []
# # for title, idx, para_idx, sentence_idx, sentence in processed_texts:
# #     org = []
# #     misc = []
# #     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
# #     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #         labels = torch.argmax(outputs.logits, dim=2)

# #         org_temp = []
# #         misc_temp = []
# #         for token, prediction in zip(tokens, labels[0].cpu().numpy()):
# #             if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
# #                 org_temp.append(token)
# #             else:
# #                 if org_temp:
# #                     org.append(" ".join(org_temp))
# #                     org_temp = []

# #             if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
# #                 misc_temp.append(token)
# #             else:
# #                 if misc_temp:
# #                     misc.append(" ".join(misc_temp))
# #                     misc_temp = []

# #     results.append((title, idx, para_idx, sentence_idx, sentence, labels, org, misc))
# #     print(results)
# #     print(len(results))  # 打印results列表的长度

# # # 将结果保存到新的CSV文件
# # result_df = pd.DataFrame(results, columns=['title','Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])
# # result_df.to_csv("result_P_A_NER.csv", index=False)

# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import re
# import torch

# # 读取Excel文件
# file_path = "NER/Updated_Results_1.xlsx"
# df = pd.read_excel(file_path, sheet_name="P-A")

# # 仅处理前10篇文章
# df = df.head(1)

# # 定义NER模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")
# model = AutoModelForTokenClassification.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")

# # 分段和句子编号函数
# def process_text(text):
#     paragraphs = text.split("\n")
#     para_sentences = []
#     para_sentence_idx = 1
#     for paragraphs_idx, para in enumerate(paragraphs):
#         para_sentence_idx = 1
#         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
#         for sentence in sentences:
#             if sentence.strip():
#                 para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip(), None, None))  # 添加两个占位符，使其具有5个值
#                 para_sentence_idx += 1
#     return para_sentences

# # 对每一行的正文列进行处理
# processed_texts = []
# for idx, row in df.iterrows():
#     text = row['正文']
#     title = row['标题']
#     para_sentences = process_text(text)
#     processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])



# # 特定段落进行测试
# test_paragraph = "Over the course of 2020, our country has suffered the devastating and unexpected loss of life due to the Covid-19 pandemic."

# # 对特定段落进行处理
# processed_test_paragraph = process_text(test_paragraph)

# # 对处理后的文本进行NER识别
# test_results = []
# for para_idx, sentence_idx, sentence, _, _ in processed_test_paragraph:
#     org_phrases = []
#     misc_phrases = []
#     org_temp = []
#     misc_temp = []
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

#     with torch.no_grad():
#         outputs = model(**inputs)
#         labels = torch.argmax(outputs.logits, dim=2)

#         for token, prediction in zip(tokens, labels[0].cpu().numpy()):
#             if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
#                 if org_temp and '##' in token:
#                     org_temp[-1] = org_temp[-1] + token[2:]
#                 else:
#                     org_temp.append(token)
#             else:
#                 if org_temp:
#                     org_phrases.append(" ".join(org_temp))
#                     org_temp = []

#             if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
#                 if misc_temp and '##' in token:
#                     misc_temp[-1] = misc_temp[-1] + token[2:]
#                 else:
#                     misc_temp.append(token)
#             else:
#                 if misc_temp:
#                     misc_phrases.append(" ".join(misc_temp))
#                     misc_temp = []

#     test_results.append((para_idx, sentence_idx, sentence, labels, org_phrases, misc_phrases))

# # 输出NER识别结果
# for result in test_results:
#     print("Sentence:", result[2])
#     print("ORG:", ", ".join(result[4]))
#     print("MISC:", ", ".join(result[5]))
#     print()

# # 将结果保存到新的CSV文件
# test_result_df = pd.DataFrame(test_results, columns=['Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])
# test_result_df.to_csv("test_result_NER.csv", index=False)






# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import re
# import torch

# # 读取Excel文件
# file_path = "/home/zsy/NER/Updated_Results_1.xlsx"
# df = pd.read_excel(file_path, sheet_name="ARTICLES")
# # 仅处理前10篇文章
# df = df.head(10)
# # 定义NER模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")
# model = AutoModelForTokenClassification.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")

# # 分段和句子编号函数
# def process_text(text):
#     paragraphs = text.split("\n")
#     para_sentences = []
#     para_sentence_idx = 1
#     for paragraphs_idx, para in enumerate(paragraphs):
#         para_sentence_idx = 1
#         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
#         for sentence in sentences:
#             if sentence.strip():
#                 para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip(), None, None))  # 添加两个占位符，使其具有5个值
#                 para_sentence_idx += 1
#     return para_sentences

# # 对每一行的正文列进行处理
# processed_texts = []
# for idx, row in df.iterrows():
#     text = row['正文']
#     title = row['标题']
#     para_sentences = process_text(text)
#     processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])

# # 对处理后的文本进行NER识别
# results = []
# for title, idx, para_idx, sentence_idx, sentence in processed_texts:
#     org_phrases = []
#     misc_phrases = []
#     org_temp = []
#     misc_temp = []
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

#     with torch.no_grad():
#         outputs = model(**inputs)
#         labels = torch.argmax(outputs.logits, dim=2)

#         for token, prediction in zip(tokens, labels[0].cpu().numpy()):
#             if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
#                 if org_temp and '##' in token:
#                     org_temp[-1] = org_temp[-1] + token[2:]
#                 else:
#                     org_temp.append(token)
#             else:
#                 if org_temp:
#                     org_phrases.append(" ".join(org_temp))
#                     org_temp = []

#             if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
#                 if misc_temp and '##' in token:
#                     misc_temp[-1] = misc_temp[-1] + token[2:]
#                 else:
#                     misc_temp.append(token)
#             else:
#                 if misc_temp:
#                     misc_phrases.append(" ".join(misc_temp))
#                     misc_temp = []

#     results.append((title, idx, para_idx, sentence_idx, sentence, labels, org_phrases, misc_phrases))

# # 输出NER识别结果
# for result in results:
#     print("Sentence:", result[4])
#     print("ORG:", ", ".join(result[6]))
#     print("MISC:", ", ".join(result[7]))
#     print()

# # 将结果保存到新的CSV文件
# result_df = pd.DataFrame(results, columns=['title', 'Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])
# result_df.to_csv("result_ARTICLES_NER_10.csv", index=False)


####################################################
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import re
# import torch

# # 读取Excel文件
# file_path = "/home/zsy/NER/Updated_Results_1.xlsx"
# df = pd.read_excel(file_path, sheet_name="ARTICLES")
# # 仅处理前10篇文章
# # df = df.head(10)
# # 定义NER模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")
# model = AutoModelForTokenClassification.from_pretrained("/home/zsy/electra-large-discriminator-finetuned-conll03-english")

# # 分段和句子编号函数
# def process_text(text):
#     paragraphs = text.split("\n")
#     para_sentences = []
#     para_sentence_idx = 1
#     for paragraphs_idx, para in enumerate(paragraphs):
#         para_sentence_idx = 1
#         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
#         for sentence in sentences:
#             if sentence.strip():
#                 para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip(), None, None))  # 添加两个占位符，使其具有5个值
#                 para_sentence_idx += 1
#     return para_sentences

# # 对每一行的正文列进行处理
# processed_texts = []
# for idx, row in df.iterrows():
#     text = row['正文']
#     title = row['标题']
#     para_sentences = process_text(text)
#     processed_texts.extend([(title, idx, para_idx, sentence_idx, sentence) for para_idx, sentence_idx, sentence, _, _ in para_sentences])

# # 对处理后的文本进行NER识别
# results = []
# for title, idx, para_idx, sentence_idx, sentence in processed_texts:
#     org_phrases = []
#     misc_phrases = []
#     org_temp = []
#     misc_temp = []
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

#     with torch.no_grad():
#         outputs = model(**inputs)
#         labels = torch.argmax(outputs.logits, dim=2)
        
#         # 迭代每个标记和预测，识别实体
#         for token, prediction in zip(tokens, labels[0].cpu().numpy()):
#             if model.config.id2label[prediction] in ["B-ORG", "I-ORG"]:
#                 if org_temp and '##' in token:
#                     org_temp[-1] = org_temp[-1] + token[2:]
#                 else:
#                     org_temp.append(token)
#             else:
#                 if org_temp:
#                     org_phrases.append(" ".join(org_temp))
#                     org_temp = []

#             if model.config.id2label[prediction] in ["B-MISC", "I-MISC"]:
#                 if misc_temp and '##' in token:
#                     misc_temp[-1] = misc_temp[-1] + token[2:]
#                 else:
#                     misc_temp.append(token)
#             else:
#                 if misc_temp:
#                     misc_phrases.append(" ".join(misc_temp))
#                     misc_temp = []

#     # 将识别结果添加到结果列表中
#     results.append((title, idx, para_idx, sentence_idx, sentence, labels, org_phrases, misc_phrases))

# # 输出NER识别结果
# for result in results:
#     print("Sentence:", result[4])
#     print("ORG:", ", ".join(result[6]))
#     print("MISC:", ", ".join(result[7]))
#     print()

# # # 将结果保存到新的CSV文件
# # result_df = pd.DataFrame(results, columns=['title', 'Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])
# # result_df.to_csv("result_ARTICLES_NER.csv", index=False)
# # 处理结果前的逻辑

# # 将结果输出到 CSV 文件之前的逻辑
# result_df = pd.DataFrame(results, columns=['title', 'Row_Index', 'Para_Index', 'Sentence_Index', 'Sentence', 'NER_Labels', 'ORG', 'MISC'])

# # 检查并修正输出的行数
# for idx, row in result_df.iterrows():
#     sentence_length = len(row['Sentence'].split())
#     org_length = sum(len(org.split()) for org in row['ORG'])
#     misc_length = sum(len(misc.split()) for misc in row['MISC'])

#     # 如果输出的长度超过了句子长度，将输出放置在下一行
#     if org_length > sentence_length or misc_length > sentence_length:
#         result_df.at[idx, 'Sentence'] = ''
#         if len(result_df) <= idx + 1:
#             result_df = result_df.append(pd.Series(), ignore_index=True)
#         result_df.at[idx + 1, 'Sentence'] = row['Sentence']

# # # 将结果保存到新的 CSV 文件
# # result_df.to_csv("result_ARTICLES_NER_corrected.csv", index=False)
# result_df.to_csv("result_ARTICLES_NER_corrected.csv", index=False, encoding='utf-8')