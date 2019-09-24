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


class Single_Pass_Cluster(object):

    def __init__(self,
                 data,
                 stop_words_file='stop_words.txt',
                 theta=0.5,
                 # LTP_DATA_DIR=r'E:\ltp-models\\ltp_data_v3.4.0\\',  # ltp模型目录的路径
                 LTP_DATA_DIR='E:\ltp_models\ltp_data_v3.4.0\ltp_data_v3.4.0',  # ltp模型目录的路径
                 segmentor=Segmentor(),
                 postagger=Postagger(),

                 ):
        self.data = data
        self.stop_words_file = stop_words_file
        self.theta = theta
        self.LTP_DATA_DIR = LTP_DATA_DIR
        self.cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
        print(self.cws_model_path)
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        self.segmentor = segmentor  # 初始化实例
        self.segmentor.load_with_lexicon(self.cws_model_path, self.LTP_DATA_DIR + 'dictionary.txt')  # 加载模型
        self.postagger = postagger  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型

    def loadData(self, data):
        Data = []
        i = 0
        # texts = [list(SentenceSplitter.split(i.strip().strip('\ufeff'))) for i in data]
        texts = data
        print('未切割前的语句总数有{}条...'.format(len(texts)))
        print("............................................................................................")
        texts = [i.strip() for i in list(_flatten(texts)) if len(i) > 5]
        print('切割后的语句总数有{}条...'.format(len(texts)))
        for text in texts:
            text = text.strip().replace('。', '').replace('」', '').replace('//', '').replace('_', '').replace('-',
                                                                                                                 '').replace(
            '\r', '').replace('\n', '').replace('\t', '').replace('@', '').replace(r'\\', '').replace("''", '')
        for line in texts:
            i += 1
            Data.append(line)

        return Data


    def word_segment(self, sentence):
        stopwords = [line.strip() for line in open(self.stop_words_file, encoding='utf-8').readlines()]
        post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b']
        sentence = sentence.strip().replace('。', '').replace('」', '').replace('//', '').replace('_', '').replace('-',
                                                                                                                 '').replace(
            '\r', '').replace('\n', '').replace('\t', '').replace('@', '').replace(r'\\', '').replace("''", '')
        words = self.segmentor.segment(sentence.replace('\n', ''))  # 分词
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
        return corpus_tfidf


    # def get_Doc2vec_vector_representation(self, word_segmentation):
    #     # 得到文本数据的空间向量表示
    #
        # corpus_doc2vec = [get_avg_feature_vector(i, model, num_features=50) for i in word_segmentation]
    #     return corpus_doc2vec


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


    def single_pass(self, corpus, texts, theta):
        dictTopic = {}
        clusterTopic = {}
        numTopic = 0
        cnt = 0

        # for vector, text in zip(corpus, texts):
        for i in np.arange(len(texts)):
            if len(corpus.corpus[i]) == 0:
                continue
            vector = corpus[i]
            text = texts[i]
            if numTopic == 0:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(text)
                numTopic += 1
            else:
                maxIndex, maxValue = self.getMaxSimilarity(dictTopic, vector)
                # 将给定语句分配到现有的、最相似的主题中
                if maxValue >= theta:
                    dictTopic[maxIndex].append(vector)
                    clusterTopic[maxIndex].append(text)

                # 或者创建一个新的主题
                else:
                    dictTopic[numTopic] = []
                    dictTopic[numTopic].append(vector)
                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].append(text)
                    numTopic += 1
            cnt += 1
            if cnt % 500 == 0:
                print("processing {}...".format(cnt))
        return dictTopic, clusterTopic


    def fit_transform(self, theta=0.5):
        datMat = self.loadData(self.data)
        word_segmentation = []
        for i in range(len(datMat)):
            word_segmentation.append(self.word_segment(datMat[i]))
        print("............................................................................................")
        print('文本已经分词完毕 !')

        # 得到文本数据的空间向量表示
        corpus_tfidf = self.get_Tfidf_vector_representation(word_segmentation)
        # corpus_tfidf =  self.get_Doc2vec_vector_representation(word_segmentation)
        dictTopic, clusterTopic = self.single_pass(corpus_tfidf, datMat, theta)
        print("............................................................................................")
        print("得到的主题数量有: {} 个 ...".format(len(dictTopic)))
        print("............................................................................................\n")
        # 按聚类语句数量对主题进行排序，找到重要的聚类群
        clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)
        for k in clusterTopic_list[:30]:
            cluster_title = '\n'.join(k[1])
            # print(''.join(cluster_title))
            # 得到每个聚类中的的主题关键词
            word = TextRank4Keyword()
            word.analyze(''.join(self.word_segment(''.join(cluster_title))), window=5, lower=True)
            w_list = word.get_keywords(num=10, word_min_len=2)
            sentence = TextRank4Sentence()
            sentence.analyze('\n'.join(k[1]), lower=True)
            s_list = sentence.get_key_sentences(num=3, sentence_min_len=5)[:30]
            print("【主题索引】:{} \n【主题声量】：{} \n【主题关键词】： {} \n【主题中心句】 ：\n{}".format(k[0], len(k[1]),
                                                                               ','.join([i.word for i in w_list]),
                                                                               '\n'.join([i.sentence for i in s_list])))
            print('\n')
            print("【标题】：",  '\n'.join([content[:20] for content in k[1]]))
            print("-------------------------------------------------------------------------")
