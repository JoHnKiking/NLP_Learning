import jieba

text = "小明毕业于北京大学计算机系"


print("精确模式")
words_generator = jieba.cut(text) # 返回一个生成器

for word in words_generator:
    print(word)

words_list = jieba.lcut(text)   # 返回一个列表
print(words_list)

print()
print("全模式")
words_generator = jieba.cut(text, cut_all=True) # 返回一个生成器
for word in words_generator:
    print(word)

words_list = jieba.lcut(text, cut_all=True) # 返回一个列表
print(words_list)

print()
print("搜索引擎模式")
words_generator = jieba.cut_for_search(text) # 返回一个生成器
for word in words_generator:
    print(word)

words_list = jieba.lcut_for_search(text) # 返回一个列表
print(words_list)

print()
print("自定义词典")
jieba.load_userdict('dict.txt')
words_list = jieba.lcut("随着云计算技术的普及，越来越多企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策。")
print(words_list)