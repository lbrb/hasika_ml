from topic_identify.single_pass_cluster import Single_Pass_Cluster
import os
import numpy as np

# single_pass_cluster = Single_Pass_Cluster('http://docs.bosonnlp.com/_downloads/text_comments.txt',
#                                           stop_words_file='stop_words.txt')

news_dir = os.walk('E:\新闻列表')
titles = []
contents = []
for path, dir_list, file_list in news_dir:
    for file in file_list:
        titles.append(file.title()[:-3])
        with open(os.path.join(path, file), encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 0:
                contents.append(' '.join(lines))
            else:
                contents.append(' ')
print(titles)
print(contents)

data = []
for i in  np.arange(len(titles)):
    content = titles[i] + contents[i].strip().replace('\n', '')
    data.append(content)

single_pass_cluster = Single_Pass_Cluster(data, stop_words_file='stop_words.txt')
single_pass_cluster.fit_transform(theta=0.25)
