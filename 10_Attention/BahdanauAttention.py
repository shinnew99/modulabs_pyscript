import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_decoder = tf.keras.layers.Dense(units)
        self.W_encoder = tf.keras.layers.Dense(units)
        self.W_combine = tf.keras.layers.Dense(1)

    def call(self, H_encoder, H_decoder):
        print("[ H_encoder ] Shape:", H_encoder.shape)

        H_encoder = self.W_encoder(H_encoder)
        print("[ W_encoder X H_encoder ] Shape:", H_encoder.shape)

        print("\n[ H_decoder ] Shape:", H_decoder.shape)
        H_decoder = tf.expand_dims(H_decoder, 1)
        H_decoder = self.W_decoder(H_decoder)

        print("[ W_decoder X H_decoder ] Shape:", H_decoder.shape)

        score = self.W_combine(tf.nn.tanh(H_decoder + H_encoder))
        print("[ Score_alignment ] Shape:", score.shape)

        attention_weights = tf.nn.softmax(score, axis=1)
        print("\n최종 Weight:\n", attention_weights.numpy())

        context_vector = attention_weights * H_decoder
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights