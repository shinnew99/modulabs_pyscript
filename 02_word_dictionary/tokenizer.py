import tensorflow as tf

class Tokenizer:
    def __init__(self):
        self.tokenizer = None

    def tokenize(self, corpus):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer.fit_on_texts(corpus)
        tensor = self.tokenizer.texts_to_sequences(corpus)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, self.tokenizer

    def print_vocabulary(self):
        print("Vocab Size:", len(self.tokenizer.index_word))
        for idx, word in enumerate(self.tokenizer.word_index):
            print(idx, ":", word)
            if idx > 10: break

