import os
import re
import json
import math
import numpy as np
from gensim import corpora, models, similarities, matutils
from smart_open import smart_open
import pandas as pd
from pyltp import SentenceSplitter
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from tkinter import _flatten
from pyltp import Segmentor, Postagger
import jieba.analyse
from topic_identify.get_docs_from_feeds import FeedsContent
from topic_identify.bean import Article, Cluster


class SinglePassCluster:
    def __init__(self, stop_words_file, user_dict_file, theta=0.40, multi_title=False, n_keywords=20):
        self.stop_words = self.get_stopwords(stop_words_file)
        self.LTP_DATA_DIR = 'E:\ltp_models\ltp_data_v3.4.0\ltp_data_v3.4.0'
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型
        self.clusters = []
        self.theta = 0.4
        self.multi_title = False
        self.n_keywords = 20
        jieba.load_userdict(user_dict_file)

    def set_params(self, theta=0.40, multi_title=False, n_keywords=20):
        self.theta = theta
        self.multi_title = multi_title
        self.n_keywords = n_keywords
        self.clusters = []

    def fit_transform(self, article):
        if article.check():
            article.pre_process(self.postagger, self.stop_words, self.n_keywords)
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
        max_sim, max_sim_cluster_id = self.get_max_similarity(article)
        cluster = None
        if max_sim > self.theta:
            cluster = self.clusters[max_sim_cluster_id]
        else:
            cluster = Cluster()
            cluster.id = len(self.clusters)
            self.clusters.append(cluster)

        cluster.add_article(article)
        article.cluster = cluster

        return cluster.id

    def get_max_similarity(self, article):
        word_tfidfs = article.word_tfidfs
        max_sim = 0
        max_sim_cluster_id = -1
        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            similarity = np.mean([matutils.cossim(article.word_tfidfs, word_tfidfs) for article in cluster.articles])
            if similarity > max_sim:
                max_sim = similarity
                max_sim_cluster_id = i

        return max_sim, max_sim_cluster_id

    def get_cluster(self):
        return self.clusters

    def show_result(self):
        feeds_content = FeedsContent()

        sorted_clusters = sorted(self.clusters, key=lambda x: len(x.articles), reverse=True)
        for i in np.arange(len(sorted_clusters)):
            cluster = sorted_clusters[i]
            if self.check_valid(cluster):
                cluster_words = set()
                for article in cluster.articles:
                    if len(cluster_words) == 0:
                        cluster_words = set(article.word_tfidfs)
                    else:
                        cluster_words = cluster_words & set(article.word_tfidfs)
                print("cluster_", i)
                print('关键词：', cluster_words)
                print('\n'.join([article.title for article in cluster.articles]))
                print('-' * 50)
            else:
                break

    def check_valid(self, cluster):
        if len(cluster.articles) > 2:
            return True
        else:
            return False