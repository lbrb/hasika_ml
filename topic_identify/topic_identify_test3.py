from topic_identify.topic_identify3 import SinglePassCluster
import os
import numpy as np
import pandas as pd
from topic_identify.pr_curve import HasikaPrCurve


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
    doc_ids = news_pd['doc_id']

    clusters = []
    groups = news_pd.groupby('聚类')
    for group in groups:
        cluster = []
        for doc_id in group[1]['doc_id']:
            cluster.append(doc_id)

        clusters.append(cluster)

    return titles, contents, clusters, doc_ids


# titles, contents = get_content_from_dir()
titles, contents, clusters, doc_ids = get_content_from_xlsx()

single_pass_cluster = SinglePassCluster(stop_words_file='stop_words.txt', user_dict_file='userdict')


def fit_transform():
    for i in np.arange(len(titles)):
        title = titles[i]
        content = contents[i]
        doc_id = doc_ids[i]

        if type(title) is str and len(title) > 3:
            # print('title_', i, title)
            single_pass_cluster.fit_transform(doc_id, title, content)
        else:
            print("title:{}, id:{}", title, i)

    single_pass_cluster.show_result()
    clusters_hat = single_pass_cluster.get_result()
    return clusters_hat


# 计算得分
pr_curve = HasikaPrCurve()

multi_arr = [True, False]
theta_arr = np.logspace(-1, 0, 20)
n_keywords_arr = np.linspace(3, 40, 40 - 2, dtype=int)

result = {}
for multi_title in multi_arr:
    for theta in theta_arr:
        for n_keywords in n_keywords_arr:
            single_pass_cluster.set_params(theta, multi_title, n_keywords)
            key_str = ':'.join([str(multi_title), str(theta), str(n_keywords)])
            print(key_str)
            clusters_hat = fit_transform()
            score = pr_curve.calc_score(clusters, clusters_hat)
            result[key_str] = score
            print(score)
            print('-' * 50)

result = sorted(result.items(), key=lambda item: item[1], reverse=True)
print(result)
