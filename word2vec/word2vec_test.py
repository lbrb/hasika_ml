import jieba
import time
from gensim.models import word2vec

time1 = time.time()
with open('天龙八部.txt', errors='ignore', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        seg_list = jieba.cut(line)
        with open('分词后的天龙八部.txt', 'a', encoding='utf-8') as ff:
            ff.write(' '.join(seg_list))
time2 = time.time()
print(time2 - time1)

sentences = word2vec.Text8Corpus('分词后的天龙八部.txt')
time3 = time.time()
print(time3 - time2)

model = word2vec.Word2Vec(sentences)
time4 = time.time()
print(time4 - time3)

for e in model.most_similar(positive=['乔峰'], topn=10):
    print(e[0], e[1])
print(time.time() - time4)
