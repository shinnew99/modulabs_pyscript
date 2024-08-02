import tensorflow as tf
from data.preprocessing import preprocess_sentence, split_spa_eng_sentences
from data.tokenizer import generate_tokenizer, make_corpus
from model.transformer import Transformer
from model.transformer import create_masks
import numpy as np

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
VOCAB_SIZE = 8000
POS_LEN = 100
MAX_LENGTH = 40

# Load and preprocess data
def load_data():
    with open('spa-eng/spa.txt', 'r', encoding='utf-8') as f:
        spa_eng_sentences = f.readlines()

    eng_sentences, spa_sentences = split_spa_eng_sentences(spa_eng_sentences)
    eng_sentences = [preprocess_sentence(sentence) for sentence in eng_sentences]
    spa_sentences = [preprocess_sentence(sentence) for sentence in spa_sentences]

    return eng_sentences, spa_sentences

eng_sentences, spa_sentences = load_data()

# Tokenization
eng_tokenizer = generate_tokenizer(eng_sentences, VOCAB_SIZE, lang="eng")
spa_tokenizer = generate_tokenizer(spa_sentences, VOCAB_SIZE, lang="spa")

# Create corpus
eng_corpus = make_corpus(eng_sentences, eng_tokenizer)
spa_corpus = make_corpus(spa_sentences, spa_tokenizer)

# Padding
eng_corpus = tf.keras.preprocessing.sequence.pad_sequences(eng_corpus, maxlen=MAX_LENGTH, padding='post')
spa_corpus = tf.keras.preprocessing.sequence.pad_sequences(spa_corpus, maxlen=MAX_LENGTH, padding='post')

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((eng_corpus, spa_corpus))
dataset = dataset.shuffle(len(eng_corpus)).batch(BATCH_SIZE, drop_remainder=True)

# Define the Transformer model
transformer = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, VOCAB_SIZE, POS_LEN, DROPOUT)

# Loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)

# Training loop
for epoch in range(EPOCHS):
    train_loss.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}')

    print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}')
