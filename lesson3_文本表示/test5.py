import jieba
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd

# 1. 加载数据
df = pd.read_csv('../Data/online_shopping_10_cats.csv', encoding='utf-8', usecols=['review'])

# 2. 查看空值位置（可选）
print(f"数据总行数: {len(df)}")
print(f"空值数量: {df['review'].isna().sum()}")
if df['review'].isna().sum() > 0:
    # 显示哪些行有空值
    na_rows = df[df['review'].isna()]
    print(f"空值所在行索引: {na_rows.index.tolist()}")
    # 如果想查看具体位置：
    for idx in na_rows.index:
        print(f"第 {idx} 行是空值")

# 3. 清理数据（关键修复！）
# 删除空值行
df_clean = df.dropna(subset=['review']).copy()
# 确保所有值都是字符串类型
df_clean['review'] = df_clean['review'].astype(str)
# 去除字符串两端的空白
df_clean['review'] = df_clean['review'].str.strip()
# 删除空字符串
df_clean = df_clean[df_clean['review'] != '']

print(f"清理后数据行数: {len(df_clean)}")

# 4. 现在可以安全分词了
sentences = []
for review in df_clean['review']:
    tokens = [token for token in jieba.lcut(review) if token.strip() != '']
    sentences.append(tokens)

print(f"成功处理 {len(sentences)} 个句子")

# 5. 训练模型
model = Word2Vec(
    sentences,          # 已分词的句子序列
    vector_size=100,    # 词向量维度
    window=5,           # 上下文窗口大小
    min_count=2,        # 最小词频（低于将被忽略）
    sg=1,               # 1:Skip-Gram，0:CBOW
    workers=4           # 并行训练线程数
)

# 6. 保存模型
model.wv.save_word2vec_format('../model/test4_vectors.kv')
print("模型保存完成！")

# 7. 加载测试
my_model = KeyedVectors.load_word2vec_format('../model/test4_vectors.kv')
print(f"模型词汇表大小: {len(my_model)}")

# 测试
test_word = '手机'
if test_word in my_model:
    print(f"'{test_word}' 在词汇表中")
    similar = my_model.most_similar(test_word, topn=3)
    print(f"与'{test_word}'最相似的词: {similar}")