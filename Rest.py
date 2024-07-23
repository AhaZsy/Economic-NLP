import pandas as pd
from fuzzywuzzy import fuzz

# 读取 Step 2 的结果文件
step2_results = pd.read_csv("/home/zsy/Article_Result_01.csv")

# 读取公司字典文件
firm_dict = pd.read_excel("/home/zsy/NER/Firms Dictionary.xlsx", sheet_name="Firm Name Clearance")

# 设置阈值
thred_entity = 90  # 实体相似度阈值
thred_firm = 50    # 公司名称相似度阈值
Score_thred = 50   # 匹配得分阈值

# 建立空列表存储结果
perfect_matches = []  # 完美匹配
excellent_matches = []  # 优秀匹配

# 相似度匹配函数
def similarity_match(entity, firm):
    entity_similarity = fuzz.token_set_ratio(entity.lower(), firm.lower())  # 计算实体相似度
    firm_similarity = fuzz.token_set_ratio(entity.lower(), firm.lower())    # 计算公司名称相似度
    return entity_similarity, firm_similarity

# 对Step 2中未匹配到的实体进行匹配
for index, row in step2_results.iterrows():
    if pd.isnull(row["Matched_Entity"]):  # 未匹配到的实体
        article_id = row["Article_ID"]
        paragraph = row["Paragraph"]
        sentence = row["Sentence"]
        entity = row["Entity"]
        industry_type = row["FFI48 (Industry type)"]

        # 进行相似度匹配
        for i, firm_row in firm_dict.iterrows():
            firm_name = firm_row["Company clear common suffix excluded"]
            entity_similarity, firm_similarity = similarity_match(entity, firm_name)
            # 测试语句：打印实体匹配情况
            print(f"Entity: {entity}, Firm: {firm_name}, Entity Similarity: {entity_similarity}, Firm Similarity: {firm_similarity}")

            # 完美匹配
            if entity_similarity == 100 or (entity_similarity > thred_entity and firm_similarity > thred_firm):
                perfect_matches.append({"Article_ID": article_id,
                                         "Paragraph": paragraph,
                                         "Sentence": sentence,
                                         "Entity": entity,
                                         "Matched_Entity": firm_name,
                                         "FFI48 (Industry type)": industry_type,
                                         "Match_Type": "Perfect Match"})
                break  # 如果找到完美匹配，跳出内循环
            # 优秀匹配
            elif entity_similarity > thred_entity and firm_similarity > thred_firm and entity_similarity > Score_thred:
                excellent_matches.append({"Article_ID": article_id,
                                           "Paragraph": paragraph,
                                           "Sentence": sentence,
                                           "Entity": entity,
                                           "Matched_Entity": firm_name,
                                           "FFI48 (Industry type)": industry_type,
                                           "Match_Type": "Excellent Match"})

# 将结果保存到 CSV 文件中
perfect_matches_df = pd.DataFrame(perfect_matches)
perfect_matches_df.to_csv("result_01.csv", index=False, encoding='utf-8-sig')

excellent_matches_df = pd.DataFrame(excellent_matches)
excellent_matches_df.to_csv("result_02.csv", index=False, encoding='utf-8-sig')
