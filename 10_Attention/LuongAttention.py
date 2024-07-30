import tensorflow as tf

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W_combine = tf.keras.layers.Dense(units)

    def call(self, H_encoder, H_decoder):
        print("[ H_encoder ] Shape:", H_encoder.shape)

        WH = self.W_combine(H_encoder)
        print("[ W_encoder X H_encoder ] Shape:", WH.shape)

        H_decoder = tf.expand_dims(H_decoder, 1)
        alignment = tf.matmul(WH, tf.transpose(H_decoder, [0, 2, 1]))
        print("[ Score_alignment ] Shape:", alignment.shape)

        attention_weights = tf.nn.softmax(alignment, axis=1)
        print("\n최종 Weight:\n", attention_weights.numpy())

        attention_weights = tf.squeeze(attention_weights, axis=-1)
        context_vector = tf.matmul(attention_weights, H_encoder)

        return context_vector, attention_weights
