from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile
from collections import defaultdict

class Tokenizer:
    def __init__(self, num_words=None, oov_token=None):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_counts = []
    def fit_on_texts(self, sentences):
        self.word_counts = defaultdict(lambda : 0)
        for sent in sentences:
            for token in sent.strip().split():
                self.word_counts[token] = self.word_counts[token] + 1

def make_w2v(input, output):
    print('시작')
    with open(input, 'rt', encoding='utf-8',) as f:
        sentences = f.readlines()

    sentences2 = [i.strip().split() for i in sentences]

    model = Word2Vec(sentences2, min_count=5, size=100, window=5,sg=1, hs=0, negative=5, iter=6, sorted_vocab=1, workers=4)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    wordcount_list = list(tokenizer.word_counts.items())
    wordcount_list.sort(key=lambda x: x[1], reverse=True)
    wordcount_list = [i[0] for i in wordcount_list if i[1]>4]

    with open(output, 'wt', encoding='utf-8') as f:
        f.write('{} {}\n'.format(len(wordcount_list), 100))
        for i in wordcount_list:
            f.write('{} {}\n'.format(i, ' '.join(map(str,model.wv[i].tolist()))))
    print('끝')

if __name__ == "__main__":
    make_w2v('renewset/en_tokened_tag_5M.txt', 'emb_en_nltk_tag.txt')
    make_w2v('renewset/kkma_all_tag.txt', 'emb_ko_kkma_tag.txt')