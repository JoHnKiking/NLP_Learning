from gensim.models import KeyedVectors

model_path = '../model/sgns.weibo.word.bz2'
model = KeyedVectors.load_word2vec_format(model_path)

print(model.vector_size)

# print(model['地铁'])

similarirty = model.similarity('地铁','公交')
print('地铁 VS 公交 相似度：', similarirty)


