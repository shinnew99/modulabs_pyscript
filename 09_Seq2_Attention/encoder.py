import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(enc_units)  # return_sequences 매개변수를 기본값 False로 전달
    
    def call(self, x):
        print("입력 Shape:", x.shape)
        x = self.embedding(x)

        print("Embedding Layer를 거친 Shape:", x.shape)
        output = self.lstm(x)

        print("LSTM Layer의 Output Shape:", output.shape)
        return output
        