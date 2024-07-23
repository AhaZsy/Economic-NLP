import pandas as pd

# 读取数据
remarks_df = pd.read_excel('/home/zsy/NER/REMARKS.xlsx')
bs_df = pd.read_excel('/home/zsy/NER/BS.xlsx')

# 创建结果数据框
results_data = []

# 遍历BS.xlsx中的标题
for i, bs_title in enumerate(bs_df['标题']):
    # 检查REMARKS.xlsx中是否存在相同的标题
    remarks_title = remarks_df[remarks_df['标题'] == bs_title]['标题'].values
    if len(remarks_title) > 0:
        # 如果存在，添加到results_data中
        results_data.append({'id': i+1, 'BS的标题': bs_title, 'Remarks的标题': remarks_title[0], '是否被包含': True})
    else:
        # 如果不存在，也添加到results_data中，但标记为未包含
        results_data.append({'id': i+1, 'BS的标题': bs_title, 'Remarks的标题': None, '是否被包含': False})

# 将结果保存到新的Excel文件中
results_df = pd.DataFrame(results_data, columns=['id', 'BS的标题', 'Remarks的标题', '是否被包含'])
results_df.to_excel('result_set.xlsx', index=False)
