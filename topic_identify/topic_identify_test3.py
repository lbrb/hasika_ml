from topic_identify.topic_identify3 import SinglePassCluster
import os
import numpy as np
import pandas as pd
from topic_identify.pr_curve import HasikaPrCurve
from topic_identify.bean import Article


class Test:
    def __init__(self):
        self.single_pass_cluster = SinglePassCluster(stop_words_file='stop_words.txt', user_dict_file='userdict')

    def cross_validate(self):
        # 计算得分
        pr_curve = HasikaPrCurve()

        multi_arr = [True, False]
        theta_arr = np.logspace(-1, 0, 20)
        n_keywords_arr = np.linspace(3, 40, 40 - 2, dtype=int)

        titles, contents, clusters, doc_ids = self.get_content_from_xlsx()

        result = {}
        for multi_title in multi_arr:
            for theta in theta_arr:
                for n_keywords in n_keywords_arr:
                    key_str = ':'.join([str(multi_title), str(theta), str(n_keywords)])
                    print(key_str)
                    clusters_hat = self.train(titles, contents, doc_ids, theta, multi_title, n_keywords)
                    score = pr_curve.calc_score(clusters, clusters_hat)
                    result[key_str] = score
                    print(score)
                    print('-' * 50)

        result = sorted(result.items(), key=lambda item: item[1], reverse=True)
        print(result)

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

        self.single_pass_cluster.show_result()
        clusters_hat = self.single_pass_cluster.get_cluster()
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


test = Test()
articles = test.get_content_from_xlsx920()
clusters_hat = test.train(articles, 0.60, False, 20)
test.save_clusters(clusters_hat)
