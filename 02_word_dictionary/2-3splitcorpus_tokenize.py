import os
import tensorflow as tf





def tokenize(corpus):  # corpus: Tokenized Sentence's List
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus)

    tensor = tokenizer.texts_to_sequences(corpus)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer



# 정제된 데이터 filtered_corpus를 공백 기반으로 토큰화하여 저장하는 코드를 직접 작성해 보세요.
split_corpus = []

for kor in filtered_corpus:
    split_corpus.append(kor.split())
    

split_tensor, split_tokenizer = tokenize(split_corpus)
print("Split Vocab Size:", len(split_tokenizer.index_word))


for idx, word in enumerate(split_tokenizer.word_index):
    print(idx, ":", word)
    if idx > 10: break