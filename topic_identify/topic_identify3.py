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


class SinglePassCluster:
    def __init__(self, stop_words_file, user_dict_file, theta=0.40, multi_title=False, n_keywords=20):
        self.stop_words = self.get_stopwords(stop_words_file)
        self.LTP_DATA_DIR = 'E:\ltp_models\ltp_data_v3.4.0\ltp_data_v3.4.0'
        self.cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load_with_lexicon(self.cws_model_path, self.LTP_DATA_DIR + 'dictionary.txt')  # 加载模型
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型
        self.post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b', 'v']
        self.titles = {}
        self.contents = {}
        self.word_tfidfs = {}
        self.clusters = []
        self.clusters_mean = []
        self.doc_id = 0
        self.theta = 0.4
        self.multi_title = False
        self.n_keywords = 20
        jieba.load_userdict(user_dict_file)

    def set_params(self, theta=0.40, multi_title=False, n_keywords=20):
        self.theta = theta
        self.multi_title = multi_title
        self.n_keywords = n_keywords
        self.titles = {}
        self.contents = {}
        self.word_tfidfs = {}
        self.clusters = []
        self.clusters_mean = []
        self.doc_id = 0

    def fit_transform(self, i, title, content):
        self.doc_id = i
        content = self.get_content(title, content)
        word_tfidfs = self.get_words(content)
        print(title)
        print(word_tfidfs)
        self.titles[self.doc_id] = title
        self.contents[self.doc_id] = content
        self.word_tfidfs[self.doc_id] = word_tfidfs
        self.single_pass(word_tfidfs)

    def predict(self, cluster_id, title, content):
        cluster = self.clusters[cluster_id]

    def get_stopwords(self, stop_words_path):
        stop_words = set()
        with open(stop_words_path, encoding='utf-8') as f:
            for line in f.readlines():
                stop_words.add(line.strip())
        return stop_words

    def get_words(self, content):
        word_tfidfs = jieba.analyse.extract_tags(content, topK=self.n_keywords, withWeight=True)
        words, tfidfs = zip(*word_tfidfs)
        words_tags = self.postagger.postag(list(words))
        word_tag_dict = dict(zip(words, words_tags))
        effective_words = [w for w, t in word_tag_dict.items() if t in self.post_list and w not in self.stop_words]
        effective_word_tfidfs = [(word, tfidf) for word, tfidf in word_tfidfs if word in effective_words]
        return effective_word_tfidfs

    def get_content(self, title, content):
        if type(content) is str and len(content) > 5:
            if self.multi_title:
                multi = max((int)(len(content) / len(title)), 1)
            else:
                multi = 1
            content = title * multi + content
        else:
            content = title

        return content

    def single_pass(self, word_tfidfs):
        max_sim, max_sim_cluster_id = self.get_max_similarity(word_tfidfs)
        if max_sim > self.theta:
            self.clusters[max_sim_cluster_id].append(self.doc_id)
        else:
            self.clusters.append([self.doc_id])

    def get_max_similarity(self, word_tfidfs):
        max_sim = 0
        max_sim_cluster_id = -1
        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            similarity = np.mean([matutils.cossim(self.word_tfidfs[doc_id], word_tfidfs) for doc_id in cluster])
            if similarity > max_sim:
                max_sim = similarity
                max_sim_cluster_id = i

        return max_sim, max_sim_cluster_id

    def get_jarcard_similarity(self, set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2))

    def get_same_num(self, set1, set2):
        # print("set(set1) & set(set2):", set(set1) & set(set2))
        return len(set(set1) & set(set2))

    def get_cosin_similarity(self, vec1, vec2):
        vector_a = np.mat(vec1)
        vector_b = np.mat(vec2)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def get_result(self):
        return self.clusters

    def show_result(self):
        feeds_content = FeedsContent()

        sorted_clusters = sorted(self.clusters, key=lambda x: len(x), reverse=True)
        for i in np.arange(len(sorted_clusters)):
            cluster = sorted_clusters[i]
            if self.check_valid(cluster):
                cluster_words = set()
                for j in cluster:
                    if len(cluster_words) == 0:
                        cluster_words = set(self.word_tfidfs[j])
                    else:
                        cluster_words = cluster_words & set(self.word_tfidfs[j])

                # 查询内容库
                # feeds_content_article = {}
                # for cluster_word in cluster_words:
                #     articles = feeds_content.getDocs(cluster_word)
                #     for article in articles:
                #         article_id = article['article_id']
                #         article_title = article['title']
                #         article_content = article['content']
                #         if article_id in feeds_content_article.keys():
                #             continue
                #         else:


                print("cluster_", i)
                print('关键词：', cluster_words)
                print('\n'.join([self.titles[j] for j in cluster]))
                print('-' * 50)
            else:
                break

    def check_valid(self, cluster):
        if len(cluster) > 2:
            return True
        else:
            return False