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
        self.post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b']
        # self.post_list = ['n','ns','j']

        self.GENSIM_DATA_DIR = 'E:\gensim_model'
        self.word2vec_model_path = os.path.join(self.GENSIM_DATA_DIR, 'baike_26g_news_13g_novel_229g.bin')
        self.word2vec = models.KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=True, limit=500000)
        self.word2vec.init_sims(replace=True)

        self.titles = {}
        self.contents = {}
        # self.words = {}
        self.word_tfidfs = {}
        self.clusters = []
        self.clusters_mean = []
        self.doc_id = 0
        self.theta = theta
        jieba.load_userdict(user_dict_file)
        self.vec = {}
        self.word2vec_error = 0
        self.word2vec_success = 0
        self.word2vec_all = 0

    def fit_transform(self, i, title, content):
        self.doc_id = i
        content = self.get_content(title, content)
        word_tfidfs = self.get_words(content)
        self.titles[self.doc_id] = title
        self.contents[self.doc_id] = content
        self.word_tfidfs[self.doc_id] = word_tfidfs
        self.single_pass(i)

    def get_words(self, content):
        word_tfidfs = jieba.analyse.extract_tags(content, topK=20, withWeight=True)
        words, tfidfs = zip(*word_tfidfs)
        words_tags = self.postagger.postag(list(words))
        word_tag_dict = dict(zip(words, words_tags))
        effective_words = [w for w, t in word_tag_dict.items() if t in self.post_list]
        # print("words:", effective_words)
        effective_word_tfidfs = [(word, tfidf) for word, tfidf in word_tfidfs if word in effective_words]

        return effective_word_tfidfs

    def get_content(self, title, content):
        multi = max((int)(len(content) / len(title)), 1)
        # multi = 1
        content = title * multi + content
        return content

    def single_pass(self, doc_id):
        max_sim, max_sim_cluster_id = self.get_max_similarity(doc_id)
        if max_sim > 0.90:
            self.clusters[max_sim_cluster_id].append(self.doc_id)
        else:
            self.clusters.append([self.doc_id])

    def get_max_similarity(self, doc_id):
        max_sim = 0
        max_sim_cluster_id = -1

        for i in np.arange(len(self.clusters)):
            cluster = self.clusters[i]

            cossims = []
            for cluster_doc_id in cluster:
                words1 = self.word_tfidfs[cluster_doc_id]
                words2 = self.word_tfidfs[doc_id]
                vec1 = self.get_vec_from_words(words1)
                vec2 = self.get_vec_from_words(words2)
                cossim = self.get_cosin_similarity(vec1, vec2)
                cossims.append(cossim)
                title1 = self.titles[cluster_doc_id]
                print('title1:', title1)
                print("words1:", words1)
                title2 = self.titles[doc_id]
                print('title2:', title2)
                print('words2:', words2)
                print('cossim:', cossim)
                print('*' * 50)
            similarity = np.mean(cossim)

            # similarity = np.mean([self.get_cosin_similarity(self.get_vec_from_words(self.word_tfidfs[doc_id]), self.get_vec_from_words(word_tfidfs)) for doc_id in cluster])
            # print("similarity:", similarity)
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

    def get_vec_from_words(self, word_tfidfs):
        word_tfidfd_vecs = np.average([self.get_vec(word) * tfidf for word, tfidf in word_tfidfs], axis=0)
        return word_tfidfd_vecs

    def get_vec(self, word):
        try:
            vec = self.word2vec[word]
        except:
            vec = np.zeros(128)
        return vec

    def show_result(self):
        sorted_clusters = sorted(self.clusters, key=lambda x: len(x), reverse=True)
        for i in np.arange(len(sorted_clusters)):
            cluster = sorted_clusters[i]
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
