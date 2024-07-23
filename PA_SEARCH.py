import pandas as pd
import re

# 分段和句子编号函数
def process_text(text):
    paragraphs = text.split("\n")
    para_sentences = []
    for paragraphs_idx, para in enumerate(paragraphs, 1):
        para_sentence_idx = 1
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
        for sentence in sentences:
            if sentence.strip():
                para_sentences.append((paragraphs_idx, para_sentence_idx, sentence.strip()))
                para_sentence_idx += 1
    return para_sentences

# 读取ARTICLES.xlsx的前100条数据
articles_df = pd.read_excel('/home/zsy/NER/PA.xlsx')
keywords_dict = pd.read_excel('/home/zsy/NER/annual keywords dictionary.xlsx', sheet_name=None)

# 为2020和2021合并处理
keywords_dict['2020~2021'] = keywords_dict['2020（2021）']

# 处理时间列
articles_df['时间'] = pd.to_datetime(articles_df['时间'], format='%Y-%m-%d %H:%M:%S')
articles_df['年份'] = articles_df['时间'].dt.year
articles_df_2020_2021 = articles_df[(articles_df['年份'] == 2020) | (articles_df['年份'] == 2021)]
articles_by_year = {
    '2017': articles_df[articles_df['年份'] == 2017],
    '2018': articles_df[articles_df['年份'] == 2018],
    '2019': articles_df[articles_df['年份'] == 2019],
    '2020~2021': articles_df_2020_2021
}

# 初始化结果列表
results = []

# 处理每一年的文章
for year, articles in articles_by_year.items():
    # 获取该年的关键词
    keywords = keywords_dict[year]
    print(keywords)
    
    # 遍历每篇文章
    for index, row in articles.iterrows():
        article_text = row['正文']
        
        # 确保 article_text 是字符串
        if not isinstance(article_text, str):
            article_text = str(article_text)
        
        # 分段和句子编号
        para_sentences = process_text(article_text)
        
        # 遍历每个段落和句子
        for para_idx, sent_idx, sentence in para_sentences:
            # 确保 sentence 是字符串
            if not isinstance(sentence, str):
                sentence = str(sentence)
            
            # 分词
            words = re.findall(r'\b\w+\b', sentence)
            
            # 查找每个行业的关键词
            for _, kw_row in keywords.iterrows():
                industry = kw_row['industry']
                keyword = kw_row['token']
                
                # 确保 keyword 是字符串
                if not isinstance(keyword, str):
                    keyword = str(keyword)
                
                # 如果关键词在句子中
                if keyword in words:
                    result = {
                        'id': row['id'],
                        '日期': row['时间'],
                        '标题': row['标题'],
                        '段落编号': para_idx,
                        '句子编号': sent_idx,
                        '文本内容': sentence,
                        '行业关键词': keyword,
                        '具体行业': industry
                    }
                    results.append(result)
                   

# 将结果转换为DataFrame并保存为Excel文件
results_df = pd.DataFrame(results)
results_df.to_excel('/home/zsy/NER/PA_word.xlsx', index=False)
