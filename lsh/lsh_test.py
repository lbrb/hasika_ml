import jieba
from sklearn.feature_extraction.text import CountVectorizer
from lsh.hasika_lsh import LSH


def load_title(path):
    titles = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            titles.append(line.strip())
    return titles


if __name__ == '__main__':
    stop_words_path = 'stop_words.txt'
    titles_path = 'titles2'
    titles = load_title(titles_path)

    title_words_list = []
    for title in titles:
        words = list(jieba.cut(title))
        # print(title)
        # print("words: ", words)
        title_words_list.append(words)

    lsh = LSH(stop_words_path)
    lsh.fit(title_words_list)
