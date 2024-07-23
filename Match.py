import pandas as pd

# 读取实体识别结果文件
ner_results = pd.read_csv("/home/zsy/NER/BS_NER_Final.csv")

# 读取公司字典文件
firm_dict = pd.read_excel("/home/zsy/NER/Firms Dictionary.xlsx", sheet_name="Firm Name Clearance")

# 将 "Company clear common suffix excluded" 列的内容转换为小写
firm_dict["Company clear common suffix excluded"] = firm_dict["Company clear common suffix excluded"].str.lower()

# 获取有效实体列表和无效实体列表
valid_entities = set(firm_dict["Company clear common suffix excluded"])

# 创建一个字典用于存储实体对应的行业类型
entity_to_industry = dict(zip(firm_dict["Company clear common suffix excluded"], firm_dict["FFI48 (Industry type)"]))

# 创建一个空列表来存储处理后的结果
processed_results_list = []

# 对每个句子的实体识别结果进行处理
for index, row in ner_results.iterrows():
    article_id = row["Row_Index"]
    paragraph = row["Para_Index"]
    sentence = row["Sentence"]
    org_entities = eval(row["ORG"]) # 
    misc_entities = eval(row["MISC"])  #
    # 合并ORG和MISC列中的实体列表
    entities = org_entities + misc_entities
    # 遍历句子中的实体
    for entity in entities:
        if entity.lower() in valid_entities:
            print(f"Valid entity found: {entity}")
            # 获取实体对应的行业类型
            industry_type = entity_to_industry.get(entity.lower(), "")
            processed_results_list.append({"Article_ID": article_id, 
                                           "Paragraph": paragraph,
                                           "Sentence": sentence,
                                           "Entity": entity,
                                           "Matched_Entity": entity,
                                           "FFI48 (Industry type)": industry_type,
                                           "Match_Type": "Perfect Match"})
        else:
            print(f"Invalid entity found: {entity}")
            processed_results_list.append({"Article_ID": article_id, 
                                           "Paragraph": paragraph,
                                           "Sentence": sentence,
                                           "Entity": entity,
                                           "Matched_Entity": "",
                                           "FFI48 (Industry type)": "",
                                           "Match_Type": "Invalid Entity"})

# 将结果列表转换为DataFrame
processed_results = pd.DataFrame(processed_results_list)

# 将结果存储到CSV文件中
processed_results.to_csv("BS_Result_01.csv", index=False, encoding='utf-8-sig')