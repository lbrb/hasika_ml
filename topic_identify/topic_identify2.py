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


class SinglePassCluster:
    def __init__(self, stop_words_file, theta):
        self.stop_words_file = stop_words_file
        self.theta = theta
        self.LTP_DATA_DIR = 'E:\ltp_models\ltp_data_v3.4.0\ltp_data_v3.4.0'
        self.cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
        print(self.cws_model_path)
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load_with_lexicon(self.cws_model_path, self.LTP_DATA_DIR + 'dictionary.txt')  # 加载模型
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型

    def fit_transform(self, data):
        titles = data['titles']
        word_segmentation = []
        for title in titles:
            word_segmentation.append(self.word_segment(title))

        # 得到文本数据的空间向量表示
        corpus_tfidf, dictionary = self.get_Tfidf_vector_representation(word_segmentation)
        dictTopicVector, clusterTopic, dictTopicWords = self.single_pass(corpus_tfidf, titles, self.theta)

        clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)
        for k in clusterTopic_list[:30]:
            words = [dictionary.__getitem__(id) for id in dictTopicWords[k[0]]]
            print(dictTopicWords[k[0]])
            print(words)
            print(k)
            print('-' * 50)

    def word_segment(self, sentence):
        stopwords = [line.strip() for line in open(self.stop_words_file, encoding='utf-8').readlines()]
        post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b']
        sentence = sentence.strip().replace('。', '').replace('」', '').replace('//', '').replace('_', '').replace('-',
                                                                                                                 '').replace(
            '\r', '').replace('\n', '').replace('\t', '').replace('@', '').replace(r'\\', '').replace("''", '')
        words = self.segmentor.segment(sentence)  # 分词
        postags = self.postagger.postag(words)  # 词性标注
        dict_data = dict(zip(words, postags))
        table = {k: v for k, v in dict_data.items() if v in post_list}
        words = list(table.keys())
        word_segmentation = []
        for word in words:
            if word == ' ':
                continue
            if word not in stopwords:
                word_segmentation.append(word)
        return word_segmentation

    def get_Tfidf_vector_representation(self, word_segmentation, pivot=10, slope=0.1):
        # 得到文本数据的空间向量表示
        dictionary = corpora.Dictionary(word_segmentation)
        corpus = [dictionary.doc2bow(text) for text in word_segmentation]
        tfidf = models.TfidfModel(corpus, pivot=pivot, slope=slope)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf, dictionary

    def getMaxSimilarity(self, dictTopic, vector):
        maxValue = 0
        maxIndex = -1
        for k, cluster in dictTopic.items():
            oneSimilarity = np.mean([matutils.cossim(vector, v) for v in cluster])
            # oneSimilarity = np.mean([cosine_similarity(vector, v) for v in cluster])
            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus_tfidf, corpus, theta):
        dictTopicVector = {}
        clusterTopic = {}
        dictTopicWords = {}
        numTopic = 0
        cnt = 0
        for i in np.arange(len(corpus)):
            if len(corpus_tfidf.corpus[i]) == 0:
                continue
            vector = corpus_tfidf[i]
            text = corpus[i]
            words = corpus_tfidf.corpus[i]
            if numTopic == 0:
                dictTopicVector[numTopic] = []
                dictTopicVector[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(text)
                dictTopicWords[numTopic] = []
                dictTopicWords[numTopic].append(words)
                numTopic += 1
            else:
                maxIndex, maxValue = self.getMaxSimilarity(dictTopicVector, vector)
                # 将给定语句分配到现有的、最相似的主题中
                if maxValue >= theta:
                    dictTopicVector[maxIndex].append(vector)
                    clusterTopic[maxIndex].append(text)
                    dictTopicWords[maxIndex].append(words)
                # 或者创建一个新的主题
                else:
                    dictTopicVector[numTopic] = []
                    dictTopicVector[numTopic].append(vector)
                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].append(text)
                    dictTopicWords[numTopic] = []
                    dictTopicWords[numTopic].append(words)
                    numTopic += 1
            cnt += 1
            if cnt % 500 == 0:
                print("processing {}...".format(cnt))
        return dictTopicVector, clusterTopic, dictTopicWords
