import jieba
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd

df = pd.read_csv('../Data/online_shopping_10_cats.csv', encoding='utf-8', usecols=['review'])

sentences = [[token for token in jieba.lcut(review) if token.strip() != ''] for review in df["review"]]



model = Word2Vec(
    sentences,          # 已分词的句子序列
    vector_size=100,    # 词向量维度
    window=5,           # 上下文窗口大小
    min_count=2,        # 最小词频（低于将被忽略）
    sg=1,               # 1:Skip-Gram，0:CBOW
    workers=4           # 并行训练线程数
)

model.wv.save_word2vec_format('../model/test4_vectors.kv')

model.wv.save_word2vec_format('../model/test4_vectors.kv')

my_model = KeyedVectors.load_word2vec_format('../model/test4_vectors.kv')
print(my_model)
