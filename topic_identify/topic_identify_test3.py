import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from topic_identify.topic_identify3 import SinglePassCluster
import os
import numpy as np
import pandas as pd
from topic_identify.pr_curve import HasikaPrCurve
from topic_identify.bean import Article
import json


class Test:
    def __init__(self):
        self.single_pass_cluster = SinglePassCluster(stop_words_file='stop_words.txt', user_dict_file='userdict')

    def cross_validate(self):
        # 计算得分
        pr_curve = HasikaPrCurve()

        multi_arr = [True, False]
        theta_arr = np.logspace(-1, 0, 20)
        n_keywords_arr = np.linspace(5, 40, 20, dtype=int)

        articles, clusters = self.get_content_from_xlsx()

        f = open('result.txt', mode='w', encoding='utf-8')
        for multi_title in multi_arr:
            for theta in theta_arr:
                for n_keywords in n_keywords_arr:
                    key_str = ' '.join([str(multi_title), str(theta), str(n_keywords)])
                    clusters_hat = self.train(articles, theta, multi_title, n_keywords)
                    cluster_ids = [[article.id for article in cluster.articles] for cluster in clusters_hat]
                    p, r = pr_curve.calc_pr(clusters, cluster_ids)
                    line = key_str + ' ' + str(p) + ' ' + str(r)
                    print(key_str, p, r)
                    f.write(line + '\n')
                    f.flush()
        f.close()

    def get_content_from_dir(self):
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

    def get_content_from_xlsx(self):
        xlsx_path = 'cluster_news.xlsx'
        news_pd = pd.read_excel(xlsx_path)

        articles = []
        clusters_dict = {}
        for index, item in news_pd.iterrows():
            article = Article()
            article.id = index
            article.title = item['新闻标题']
            article.content = item['正文内容']
            articles.append(article)

            cluster_id = item['聚类']
            if cluster_id in clusters_dict.keys():
                clusters_dict[cluster_id].append(index)
            else:
                clusters_dict[cluster_id] = [index]

        clusters_dict = list(clusters_dict.values())

        return articles, clusters_dict

    def get_content_from_xlsx920(self):
        xlsx_path = 'output_09_24_simple.xls'
        news_df = pd.read_excel(xlsx_path)

        articles = []
        for index, item in news_df.iterrows():
            article = Article()
            article.id = index
            article.title = item['新闻标题']
            article.content = item['正文内容']
            articles.append(article)

        return articles

    def train(self, articles, theta, multi_title, n_keywords):
        self.single_pass_cluster.set_params(theta, multi_title, n_keywords)

        for i in np.arange(len(articles)):
            self.single_pass_cluster.fit_transform(articles[i])

        # self.single_pass_cluster.show_result()
        clusters_hat = self.single_pass_cluster.get_clusters()
        return clusters_hat

    def save_clusters(self, clusters_hat):
        xlsx_path = 'output_09_24_simple.xls'
        news_df = pd.read_excel(xlsx_path)

        for cluster in clusters_hat:
            for article in cluster.articles:
                doc_id = article.id
                cluster_id = article.cluster.id
                news_df.loc[doc_id, '聚类'] = cluster_id

        news_df.to_excel('cluster_news.xls', encoding='utf-8')

    def get_cluster_id(self, clusters, doc_id):
        for i in np.arange(len(clusters)):
            if doc_id in clusters[i]:
                return i
        else:
            return -1

    def run(self):
        articles = self.get_content_from_xlsx920()
        clusters_hat = self.train(articles, 0.545, True, 20)
        # self.save_clusters(clusters_hat)


if __name__ == '__main__':
    test = Test()
    test.cross_validate()
    # test.run()
