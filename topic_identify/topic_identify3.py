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


class SinglePassCluster:
    def __init__(self, stop_words_file, user_dict_file, theta):
        self.stop_words_file = stop_words_file
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
        self.words = {}
        self.clusters = []
        self.cur_i = 0
        self.theta = theta
        jieba.load_userdict(user_dict_file)

    def fit_transform(self, i, title, content):
        self.cur_i = i
        content = self.get_content(title, content)
        words = self.get_words(content)
        # print(title)
        # print(words)
        self.titles[self.cur_i] = title
        self.contents[self.cur_i] = content
        self.words[self.cur_i] = words
        self.single_pass(words)

    def get_words(self, content):
        words = jieba.analyse.extract_tags(content, topK=20)
        words_tags = self.postagger.postag(words)
        word_tag_dict = dict(zip(words, words_tags))
        # print("word_tag_dict:", word_tag_dict)
        words3 = [w for w, t in word_tag_dict.items() if t in self.post_list]
        return words3

    def get_content(self, title, content):
        multi = max((int)(len(content) / len(title)), 1)
        content = title * multi + content
        return content

    def single_pass(self, words):
        has_cluster = False
        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            mean = np.mean([self.get_same_num(self.words[index], words) for index in cluster])
            # print("mean: ", mean)
            if mean > 2:
                cluster.append(self.cur_i)
                has_cluster = True
                break
        if not has_cluster:
            self.clusters.append([self.cur_i])

    def get_jarcard_similarity(self, set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2))

    def get_same_num(self, set1, set2):
        # print("set(set1) & set(set2):", set(set1) & set(set2))
        return len(set(set1) & set(set2))

    def show_result(self):
        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]
            print("cluster_", i)
            cluster_words = set()
            for j in cluster:
                if len(cluster_words) == 0:
                    cluster_words = set(self.words[j])
                else:
                    cluster_words = cluster_words & set(self.words[j])

            print('关键词：', cluster_words)
            print('\n'.join([self.titles[j] for j in cluster]))
            print('-' * 50)
