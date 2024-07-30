import tensorflow as tf

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences = True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x, context_v):
        print("입력 Shape:", x.shape)
        x = self.embedding(x)
        print("Embedding Layer를 거친 Shape:", x.shape)

        context_v = tf.repeat(tf.expand_dims(context_v, axis=1), repeats = x.shape[1], axis = 1)
        x = tf.concat([x, context_v], axis=-1)
        print("Context Vector가 더해진 Shape:", x.shape)

        x = self.lstm(x)
        print("LSTM Layer의 Output Shape:", x.shape)

        output = self.fc(x)
        print("Decoder 최종 Output Shape:", output.shape)

        return self.softmax(output)