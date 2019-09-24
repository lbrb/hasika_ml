from topic_identify.topic_identify4 import SinglePassCluster
import os
import numpy as np
import pandas as pd


def get_content_from_dir():
    # news_dir = os.walk('E:\新闻列表')
    news_dir = os.walk('news')
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
    return titles, contents


def get_content_from_xlsx():
    xlsx_path = '爬取字段9.20.xlsx'
    news_pd = pd.read_excel(xlsx_path)
    titles = news_pd['标题']
    contents = news_pd['正文内容']
    return titles, contents


# titles, contents = get_content_from_dir()
titles, contents = get_content_from_xlsx()

single_pass_cluster = SinglePassCluster(stop_words_file='stop_words.txt', user_dict_file='userdict', theta=0.16)
content_dict = dict(zip(titles, contents))
for i in np.arange(len(content_dict.items())):
    title, content = list(content_dict.items())[i]
    if type(title) is str and len(title) > 3:
        if '断交' in content or '断交' in title:
            single_pass_cluster.fit_transform(i, title, content[:200])

single_pass_cluster.show_result()
