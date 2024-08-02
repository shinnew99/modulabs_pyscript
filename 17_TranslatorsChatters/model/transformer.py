import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense, Embedding
from tensorflow.keras import layers


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
    enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask
    dec_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
    dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask
    size = tf.shape(tar)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    combined_mask = tf.maximum(look_ahead_mask, 0)

    return enc_padding_mask, combined_mask, dec_padding_mask

class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ff, src_vocab_size, tgt_vocab_size, pos_len, dropout=0.2, shared_fc=True):
        super(Transformer, self).__init__()
        self.shared_fc = shared_fc
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, src_vocab_size, pos_len, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, tgt_vocab_size, pos_len, dropout)
        self.final_layer = Dense(tgt_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size, pos_len, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.pos_len = pos_len
        self.embedding = Embedding(vocab_size, d_model)
        self.position_encoding = self.positional_encoding(pos_len, d_model)
        self.dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]

    def positional_encoding(self, pos_len, d_model):
        position = tf.range(pos_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pos_enc = tf.sin(position * div_term)
        pos_enc = tf.concat([pos_enc, tf.cos(position * div_term)], axis=-1)
        return pos_enc[tf.newaxis, :]

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.layers:
            x = layer(x, training, mask)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size, pos_len, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.pos_len = pos_len
        self.embedding = Embedding(vocab_size, d_model)
        self.position_encoding = self.positional_encoding(pos_len, d_model)
        self.dropout = Dropout(dropout)
        self.layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]

    def positional_encoding(self, pos_len, d_model):
        position = tf.range(pos_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pos_enc = tf.sin(position * div_term)
        pos_enc = tf.concat([pos_enc, tf.cos(position * div_term)], axis=-1)
        return pos_enc[tf.newaxis, :]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.layers:
            x = layer(x, enc_output, training, look_ahead_mask, padding_mask)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.attention_layernorm = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_model)
        ])
        self.ffn_layernorm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

    def call(self, x, training, mask):
        attn_output = self.attention(x, x, attention_mask=mask)
        attn_output = self.dropout(attn_output, training=training)
        x = self.attention_layernorm(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, training=training)
        x = self.ffn_layernorm(x + ffn_output)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attention1 = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.attention2 = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.attention1_layernorm = LayerNormalization(epsilon=1e-6)
        self.attention2_layernorm = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_model)
        ])
        self.ffn_layernorm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.attention1(x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout(attn1, training=training)
        x = self.attention1_layernorm(x + attn1)
        attn2 = self.attention2(x, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout(attn2, training=training)
        x = self.attention2_layernorm(x + attn2)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, training=training)
        x = self.ffn_layernorm(x + ffn_output)
        return x
