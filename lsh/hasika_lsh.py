import numpy as np


class LSH:
    def __init__(self, stopwords_path):
        self.stopwords = self.load_stopwords(stopwords_path)

    def fit(self, X):
        samples = self.remove_stopwords(X)
        words = list(set([w for a in samples for w in a]))
        indexes = np.arange(0, len(words))
        print(words)
        print(indexes)
        self.words_dict = dict(zip(words, indexes))
        print(self.words_dict)

        tokenizes = []
        for sample in samples:
            print(sample)
            tokenize = self.gen_tokenize(sample)
            print(tokenize)
            tokenizes.append(tokenize)
        print(tokenizes)

    def gen_tokenize(self, sample):
        tokenize = np.zeros(len(self.words_dict))
        for w in sample:
            tokenize[self.words_dict[w]] = 1
        return tokenize

    def remove_stopwords(self, X):
        samples = []
        for sample in X:
            sample = [w for w in sample if w not in self.stopwords]
            samples.append(sample)
        return samples

    def load_stopwords(self, path):
        stop_words = set()
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                stop_words.add(line.strip())
        stop_words.add(' ')
        return stop_words


