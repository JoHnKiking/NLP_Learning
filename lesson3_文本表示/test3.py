from gensim.models import Word2Vec


sentences = [['我', '每天','乘坐', '地铁', '上班'], ['我','每天', '乘坐', '公交', '上班']]

model = Word2Vec(
    sentences,          # 已分词的句子序列
    vector_size=100,    # 词向量维度
    window=5,           # 上下文窗口大小
    min_count=2,        # 最小词频（低于将被忽略）
    sg=1,               # 1:Skip-Gram，0:CBOW
    workers=4           # 并行训练线程数
)

model.wv.save_word2vec_format('../model/my_vectors.kv')