import jieba.analyse


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
            for word in article.effective_words:
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
        self.multi_title = False
        self.effective_words = []
        self.effective_word_tfidfs = []
        self.effective_word_posts = []
        self.cluster = None
        self.post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b', 'v']

    def pre_process(self, postagger, stop_words, n_keywords):
        content = self.get_content()
        word_tfidfs = jieba.analyse.extract_tags(content, topK=n_keywords, withWeight=True)
        words, tfidfs = zip(*word_tfidfs)
        words_tags = postagger.postag(list(words))
        word_tag_dict = dict(zip(words, words_tags))

        self.effective_words = [w for w, t in word_tag_dict.items() if t in self.post_list and w not in stop_words]
        self.effective_word_tfidfs = [(word, tfidf) for word, tfidf in word_tfidfs if word in self.effective_words]
        self.effective_word_posts = [(word, postag) for word, postag in word_tag_dict.items() if
                                     word in self.effective_words]
        #
        # print(self.title)
        # print(word_tfidfs)
        # print(self.effective_word_tfidfs)
        # print(word_tag_dict)
        # print(self.effective_word_posts)

    def get_content(self):
        if self.multi_title:
            if type(self.content) is str and len(self.content) > 5:
                multi = max((int)(len(self.content) / len(self.title)), 1)
                return self.title * multi + self.content
            else:
                return self.title
        elif type(self.content) is not str:
            return self.title
        else:
            return self.title + self.content

    def check(self):
        if type(self.title) is str:
            return True
        else:
            return False
