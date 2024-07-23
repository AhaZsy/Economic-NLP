# Economic-NLP
1.模型去hugging face下载electra-large-discriminator-finetuned-conll03-english，用于实体提取
2.其他python文件主要做以下工作
1.首先完成初步过滤，得到Excel文件
2.远程服务器（csdn)  ，待项目完成之后，会写个博客记录
点开远程那个图标  然后设置的符号 ，去设置账号密码，还有ip地址，端口啥的..忘记了
3.
cd /
cd home文件夹 
通过linux命令下载huggingface模型

4.ctrl+链接  打开项目

5.大体步骤
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 白宫文章爬取
1 +++
工具：Selemium or BeautifulSoup
地址：https://trumpwhitehouse.archives.gov/news/ 
内容：文章属性-标题-时间-正文
2. 文章维度与特朗普无关的过滤 - president trump
3. 文章级别NER识别：公司&组织位置
3 +++
首先先做段落拆分，对原有的excel文件的正文那一列，按照段落进行处理，每个段落按小于等于512个token处理，然后再进行ner识别
（对原有的excel文件的正文那一列，先按照段落进行处理（先分段做编号），然后每个段落再按句子分(给句子编号），然后基于句子再进行ner识别
Step 1.  对处理后的text采用 ner-english-large 抽取实体 ORG 和 MISC
Step 2. 根据有效entity 和 无效entity对应表，预处理Step1结果，将单个句子所有entity处理完的放在result_{id}_01.csv中. (认为其高准)
Step 3.  Step2剩下的，通过相似度匹配算法匹配公司，thred_entity = 0.9, thred_firm = 0.5, Score_thred = 0.5。处理完的结果放在result_{id}_02.csv中
Step 4. 书剑检查两份结果，对 有效entity 和 无效entity对应表 进行响应的修改
Step 5. 根据Step1结果，只对有效entity进行对应，part1列放抽取到entity对应到的Firm， part2列放回溯找到的Firm，存放于result_{id}_03.csv中
Step 6. 书剑检查Step5结果
Step 7. 根据反馈，加上后缀， 形成最终结果
根据新条件：
使用模糊匹配，匹配程度低于90%的视为不匹配。
将完美匹配的结果（匹配分数为100%）放入一个Excel文件。
将匹配分数在90%~99%之间的结果放入另一个Excel文件。
呈现形式为：公司-匹配文本-匹配分数-所在文章信息-所在文章段落
