from fuzzywuzzy import fuzz
import pandas as pd

# Load data
article_data = pd.read_csv("/home/zsy/NER/P_A_Result_01.csv")
company_data = pd.read_excel("/home/zsy/NER/Update Dictionary.xlsx")


# Define thresholds
thred_entity = 80  # 实体相似度阈值
thred_firm = 50  # 公司名称相似度阈值
score_thred = 50  # 匹配得分阈值

# Function to calculate similarity between entity and company name
def calculate_similarity(entity, company):
    return fuzz.token_sort_ratio(entity, company)  # 直接返回匹配得分，为0到100之间的整数

# Match entities with company names
perfect_matches = []
excellent_matches = []

for index, row in article_data.iterrows():
    if row['Match_Type'] == 'Invalid Entity':
        entity = row['Entity']
        entity_id = row['Article_ID']
        paragraph = row['Paragraph']
        sentence = row['Sentence']
        matched_entities = []
        print(f"Processing entity: {entity} (Article ID: {entity_id})")
        for _, comp_row in company_data.iterrows():
            company = comp_row['Company clear common suffix excluded']
            company_ffi48 = comp_row['FFI48 (Industry type)']
            firm_similarity = calculate_similarity(entity, company)
            print(firm_similarity)
            if firm_similarity == 100:  # 如果匹配得分为100，即完美匹配
                perfect_matches.append({
                    'Article ID': entity_id,
                    'Paragraph': paragraph,
                    'Sentence': sentence,
                    'Entity': entity,
                    'Matched_Entity': company,
                    'Match_Type': 'Perfect Match',
                    'Score': firm_similarity
                })
                break  # 如果有完美匹配，则停止匹配，不再继续查找其他公司
            elif firm_similarity > thred_entity and firm_similarity > score_thred:
                excellent_matches.append({
                    'Article ID': entity_id,
                    'Paragraph': paragraph,
                    'Sentence': sentence,
                    'Entity': entity,
                    'Matched_Entity': company,
                    'Match_Type': 'Excellent Match',
                    'Score': firm_similarity
                })

# Filter results based on the specified conditions
result_01_df = pd.DataFrame(perfect_matches)
result_02_df = pd.DataFrame(excellent_matches)
print(result_01_df)
print(result_02_df)

# Data cleaning: replace commas with semicolons in entity and matched_entity columns
result_01_df['Entity'] = result_01_df['Entity'].str.replace(',', ';')
result_01_df['Matched_Entity'] = result_01_df['Matched_Entity'].str.replace(',', ';')
result_02_df['Entity'] = result_02_df['Entity'].str.replace(',', ';')
result_02_df['Matched_Entity'] = result_02_df['Matched_Entity'].str.replace(',', ';')

# Save results to CSV
result_01_df.to_csv("result_01_PA.csv", index=False, encoding='utf-8-sig')
result_02_df.to_csv("result_02_PA.csv", index=False, encoding='utf-8-sig')
