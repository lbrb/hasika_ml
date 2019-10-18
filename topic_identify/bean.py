import jieba.analyse
import numpy as np


class Cluster:
    def __init__(self):
        self.id = None
        self.articles = []
        self.keywords = []
        self.similarity_articles = []

    def add_article(self, article):
        self.articles.append(article)

    def get_important_words(self):
        cluster_words = {}
        for article in self.articles:
            for word in article.title_content_effective_words:
                if word in cluster_words.keys():
                    cluster_words[word] += 1
                else:
                    cluster_words[word] = 1

        sorted_words = sorted(cluster_words.items(), key=lambda item: item[1], reverse=True)

        return sorted_words


class Article:
    def __init__(self):
        self.id = None
        self.title = None
        self.content = None
        self.title_content_effective_words = []
        self.title_content_effective_word_tfidfs = []
        self.title_content_effective_word_posts = []
        self.title_effective_word_words = []
        self.title_effective_word_tfidfs = []
        self.title_effective_word_posts = []
        self.content_effective_word_words = []
        self.content_effective_word_tfidfs = []
        self.content_effective_word_posts = []
        self.cluster = None
        self.post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b', 'v', 'q']

    def pre_process(self, postagger, stop_words, n_keywords, multi_title):
        # title_content
        title_content = self.get_title_content(multi_title)
        title_content_word_tfidfs = jieba.analyse.extract_tags(title_content, topK=n_keywords, withWeight=True)
        title_content_words, title_content_tfidfs = zip(*title_content_word_tfidfs)
        title_content_words_tags = postagger.postag(list(title_content_words))
        title_content_word_tag_dict = dict(zip(title_content_words, title_content_words_tags))

        self.title_content_effective_words = [w for w, t in title_content_word_tag_dict.items() if
                                              t in self.post_list and w not in stop_words]
        self.title_content_effective_word_tfidfs = [(word, tfidf) for word, tfidf in title_content_word_tfidfs if
                                                    word in self.title_content_effective_words]
        self.title_content_effective_word_posts = [(word, postag) for word, postag in
                                                   title_content_word_tag_dict.items() if
                                                   word in self.title_content_effective_words]

        print(self.title)
        print(self.title_content_effective_words)
        print(self.title_content_effective_word_tfidfs)
        print(self.title_content_effective_word_posts)
        print("-" * 40)

        # title
        title_word_tfidfs = jieba.analyse.extract_tags(self.title, topK=n_keywords, withWeight=True)
        title_words, title_tfidfs = zip(*title_word_tfidfs)
        title_words_tags = postagger.postag(list(title_words))
        title_word_tag_dict = dict(zip(title_words, title_words_tags))

        self.title_effective_words = [w for w, t in title_word_tag_dict.items() if
                                      t in self.post_list and w not in stop_words]
        self.title_effective_word_tfidfs = [(word, tfidf) for word, tfidf in title_word_tfidfs if
                                            word in self.title_effective_words]
        self.title_effective_word_posts = [(word, postag) for word, postag in title_word_tag_dict.items() if
                                           word in self.title_effective_words]

        # content
        content_word_tfidfs = jieba.analyse.extract_tags(self.content, topK=n_keywords, withWeight=True)
        content_words, content_tfidfs = zip(*content_word_tfidfs)
        content_words_tags = postagger.postag(list(content_words))
        content_word_tag_dict = dict(zip(content_words, content_words_tags))

        self.content_effective_words = [w for w, t in content_word_tag_dict.items() if
                                        t in self.post_list and w not in stop_words]
        self.content_effective_word_tfidfs = [(word, tfidf) for word, tfidf in content_word_tfidfs if
                                              word in self.content_effective_words]
        self.content_effective_word_posts = [(word, postag) for word, postag in content_word_tag_dict.items() if
                                             word in self.content_effective_words]

    def get_title_content(self, multi_title):
        if multi_title:
            if type(self.content) is str and len(self.content) > 5:
                log2 = np.log2((len(self.content) / len(self.title)))
                multi = int(max(log2, 1))
                return self.title * multi + self.content
            else:
                return self.title
        elif type(self.content) is not str:
            return self.title
        else:
            return self.title + self.content

    def check(self):
        if type(self.title) is str and type(self.content) is str:
            return True
        else:
            return False
