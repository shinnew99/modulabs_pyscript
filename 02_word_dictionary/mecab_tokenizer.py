from konlpy.tag import Mecab
from tokenizer import Tokenizer

class MecabTokenizer:
    def __init__(self):
        self.mecab = Mecab()

    def mecab_split(self, sentence):
        return self.mecab.morphs(sentence)

    def tokenize(self, corpus):
        mecab_corpus = [self.mecab_split(kor) for kor in corpus]
        tokenizer = Tokenizer()
        tensor, tokenizer = tokenizer.tokenize(mecab_corpus)
        return tensor, tokenizer

    def print_vocabulary(self, tokenizer):
        print("MeCab Vocab Size:", len(tokenizer.index_word))