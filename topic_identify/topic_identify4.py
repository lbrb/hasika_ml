import itertools
import os

import jieba.analyse
import numpy as np
from gensim import matutils
from pyltp import Postagger

from topic_identify.bean import Cluster
from topic_identify.feeds_content import FeedsContent


class SinglePassCluster:
    def __init__(self, stop_words_file, user_dict_file):
        self.stop_words = self.get_stopwords(stop_words_file)
        self.LTP_DATA_DIR = 'E:\ltp_models\ltp_data_v3.4.0\ltp_data_v3.4.0'
        if not os.path.exists(self.LTP_DATA_DIR):
            self.LTP_DATA_DIR = '/home/jiaolei/nlp/ltp/ltp_data_v3.4.0'
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型
        self.clusters = []
        self.title_theta = 0.4
        self.content_theta = 0.4
        self.n_keywords = 20
        jieba.load_userdict(user_dict_file)

    def set_params(self, title_theta, content_theta):
        self.title_theta = title_theta
        self.content_theta = content_theta
        self.clusters = []

    def fit_transform(self, article):
        if article.check():
            article.pre_process(self.postagger, self.stop_words, self.n_keywords, False)
            return self.single_pass(article)
        else:
            return None

    def predict(self, cluster_id, title, content):
        cluster = self.clusters[cluster_id]

    def get_stopwords(self, stop_words_path):
        stop_words = set()
        with open(stop_words_path, encoding='utf-8') as f:
            for line in f.readlines():
                stop_words.add(line.strip())
        return stop_words

    def single_pass(self, article):
        title_content_max_sim, title_content_max_sim_cluster_id, content_max_sim, content_max_sim_cluster_id, title_max_sim, title_max_sim_cluster_id = self.get_max_similarity(article)
        cluster = None
        if title_max_sim > self.title_theta or content_max_sim > self.content_theta:
            cluster = self.clusters[title_max_sim_cluster_id]
        else:
            cluster = Cluster()
            cluster.id = len(self.clusters)
            self.clusters.append(cluster)

        cluster.add_article(article)
        article.cluster = cluster

        return cluster.id

    def get_max_similarity(self, article):
        title_content_word_tfidfs = article.title_content_effective_word_tfidfs
        title_content_max_sim = 0
        title_content_max_sim_cluster_id = -1

        content_word_tfidfs = article.content_effective_word_tfidfs
        content_max_sim = 0
        content_max_sim_cluster_id = -1

        title_word_tfidfs = article.title_effective_word_tfidfs
        title_max_sim = 0
        title_max_sim_cluster_id = -1

        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            # title_content
            title_content_similarity = np.mean(
                [matutils.cossim(article.title_content_effective_word_tfidfs, title_content_word_tfidfs) for article in cluster.articles])
            if title_content_similarity > title_content_max_sim:
                title_content_max_sim = title_content_similarity
                title_content_max_sim_cluster_id = i

            content_similarity = np.mean(
                [matutils.cossim(article.content_effective_word_tfidfs, content_word_tfidfs) for article in cluster.articles])
            if content_similarity > content_max_sim:
                content_max_sim = content_similarity
                content_max_sim_cluster_id = i

            title_similarity = np.mean(
                [matutils.cossim(article.title_effective_word_tfidfs, title_word_tfidfs) for article in cluster.articles])
            if title_similarity > title_max_sim:
                title_max_sim = title_similarity
                title_max_sim_cluster_id = i


        return title_content_max_sim, title_content_max_sim_cluster_id, content_max_sim, content_max_sim_cluster_id, title_max_sim, title_max_sim_cluster_id

    def get_similarity_for_article_and_cluster(self, article, cluster):
        article.pre_process(self.postagger, self.stop_words, self.n_keywords)
        word_tfidfs = article.effective_word_tfidfs
        similarity = np.mean(
            [matutils.cossim(article.effective_word_tfidfs, word_tfidfs) for article in cluster.articles])

        return similarity

    def get_clusters(self):
        return self.clusters

    def show_result(self):
        sorted_clusters = sorted(self.clusters, key=lambda x: len(x.articles), reverse=True)
        for i in np.arange(len(sorted_clusters)):
            cluster = sorted_clusters[i]
            # self.get_most_similarity_article(cluster)
            # print("cluster_", i)
            # print('关键词：', cluster.get_important_words())
            # print('\n'.join([article.title for article in cluster.articles]))
            # print('内容库相关文章：')
            # print('\n'.join([article.title + str(similarity) for article, similarity in cluster.similarity_articles]))
            # print('-' * 50)

    def get_most_similarity_article(self, cluster):
        feeds_content = FeedsContent()
        sorted_words = cluster.get_important_words()
        effective_words = [w for w, count in sorted_words if count > 1]
        article_similarity = {}
        articles = feeds_content.get_articles([' '.join(effective_words)], 100)

        for article in articles:
            if article not in article_similarity.keys():
                article_similarity[article] = self.get_similarity_for_article_and_cluster(article, cluster)

        article_similarity = sorted(article_similarity.items(), key=lambda item: item[1], reverse=True)[:5]
        cluster.similarity_articles = article_similarity

    def check_valid(self, cluster):
        if len(cluster.articles) > 2:
            return True
        else:
            return False
