import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from preprocessing.text_preprocessing import TextPreprocessing
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2SeqModel
from utils import tokenize, loss_function
from tqdm import tqdm
import random

def main():
    print(tf.__version__)

    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    
    path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

    with open(path_to_file, "r") as f:
        raw = f.read().splitlines()

    enc_corpus = []
    dec_corpus = []

    num_examples = 30000
    for pair in raw[:num_examples]:
        eng, spa = pair.split("\t")
        enc_corpus.append(TextPreprocessing.preprocess_sentence(eng))
        dec_corpus.append(TextPreprocessing.preprocess_sentence(spa, s_token=True, e_token=True))

    enc_tensor, enc_tokenizer = tokenize(enc_corpus)
    dec_tensor, dec_tokenizer = tokenize(dec_corpus)

    enc_train, enc_val, dec_train, dec_val = train_test_split(enc_tensor, dec_tensor, test_size=0.2)

    vocab_size = len(enc_tokenizer.word_index) + 1
    emb_dim = 256
    enc_units = 512
    dec_units = 512
    BATCH_SIZE = 64
    EPOCHS = 10

    encoder = Encoder(vocab_size, emb_dim, enc_units)
    decoder = Decoder(vocab_size, emb_dim, dec_units)
    optimizer = tf.keras.optimizers.Adam()
    seq2seq = Seq2SeqModel(encoder, decoder, optimizer, loss_function)

    for epoch in range(EPOCHS):
        total_loss = 0
        idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
        random.shuffle(idx_list)
        t = tqdm(idx_list)

        for (batch, idx) in enumerate(t):
            batch_loss = seq2seq.train_step(
                enc_train[idx:idx+BATCH_SIZE],
                dec_train[idx:idx+BATCH_SIZE],
                dec_tokenizer)
    
            total_loss += batch_loss
            t.set_description_str('Epoch %2d' % (epoch + 1))
            t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))

if __name__ == "__main__":
    main()
