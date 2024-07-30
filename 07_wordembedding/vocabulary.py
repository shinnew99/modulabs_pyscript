from collections import Counter

class Vocabulary:
    def __init__(self, tokens, vocab_size = None):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.vocab = Counter(tokens)
        if vocab_size:
            self.vocab - self.vocab.most_common(vocab_size)
        self.word2idx = {word[0]: index + 1 for index, word in enumerate(self.vocab)}

    
    def get_word_index(self):
        return self.word2idx
    
    def one_hot_encoding(self, word):
        one_hot_vector = [0]*len(self.word2dix)
        index = self.word2idx[word]
        one_hot_vector[index-1] = 1
        return one_hot_vector