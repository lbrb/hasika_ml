import os
import re
import json
import math
import numpy as np
from gensim import corpora, models, similarities, matutils
from smart_open import smart_open
import pandas as pd
# from pyltp import SentenceSplitter
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from tkinter import _flatten
# from pyltp import Segmentor, Postagger
import jieba.analyse


class SinglePassCluster:
    def __init__(self, stop_words_file, user_dict_file, theta):
        self.stop_words_file = stop_words_file
        self.LTP_DATA_DIR = 'E:\ltp_models\ltp_data_v3.4.0\ltp_data_v3.4.0'
        self.cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        # self.segmentor = Segmentor()  # 初始化实例
        # self.segmentor.load_with_lexicon(self.cws_model_path, self.LTP_DATA_DIR + 'dictionary.txt')  # 加载模型
        # self.postagger = Postagger()  # 初始化实例
        # self.postagger.load(self.pos_model_path)  # 加载模型
        self.post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b', 'v']
        self.titles = {}
        self.contents = {}
        # self.words = {}
        self.word_tfidfs = {}
        self.clusters = []
        self.doc_id = 0
        self.theta = theta
        jieba.load_userdict(user_dict_file)

    def fit_transform(self, i, title, content):
        self.doc_id = i
        content = self.get_content(title, content)
        word_tfidfs = self.get_words(content)
        # print(title)
        # print(words)
        self.titles[self.doc_id] = title
        self.contents[self.doc_id] = content
        # self.words[self.cur_i] = [word_tfidfs
        self.word_tfidfs[self.doc_id] = word_tfidfs
        self.single_pass(word_tfidfs)

    def get_words(self, content):
        word_tfidfs = jieba.analyse.extract_tags(content, topK=20, withWeight=True)
        # words, tfidfs = zip(*word_tfidfs)
        # words_tags = self.postagger.postag(list(words))
        # word_tag_dict = dict(zip(words, words_tags))
        # print("word_tag_dict:", word_tag_dict)
        # effective_words = [w for w, t in word_tag_dict if t in self.post_list]
        # effective_word_tfidfs = [tuple(word, tfidf) for word, tfidf in word_tfidfs if word in effective_words]
        return word_tfidfs

    def get_content(self, title, content):
        # multi = max((int)(len(content) / len(title)), 1)
        multi = 1
        content = title * multi + content
        return content

    def single_pass(self, word_tfidfs):
        has_cluster = False
        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            mean = np.mean([matutils.cossim(self.word_tfidfs[doc_id], word_tfidfs) for doc_id in cluster])
            # mean = np.mean([self.get_cosin_similarity(self.word_tfidfs[doc_id], word_tfidfs) for doc_id in cluster])
            # print("mean: ", mean)
            if mean > 0.34:
                cluster.append(self.doc_id)
                has_cluster = True
                break
        if not has_cluster:
            self.clusters.append([self.doc_id])

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

    def show_result(self):
        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            print("cluster_", i)
            cluster_words = set()
            for j in cluster:
                if len(cluster_words) == 0:
                    cluster_words = set(self.word_tfidfs[j])
                else:
                    cluster_words = cluster_words & set(self.word_tfidfs[j])

            print('关键词：', cluster_words)
            print('\n'.join([self.titles[j] for j in cluster]))
            print('-' * 50)
