from topic_identify.topic_identify2 import SinglePassCluster
import os
import numpy as np

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

data = {'titles': titles, 'contents': contents}

single_pass_cluster = SinglePassCluster(stop_words_file='stop_words.txt', theta=0.16)
single_pass_cluster.fit_transform(data)
