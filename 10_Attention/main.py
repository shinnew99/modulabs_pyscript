# from BahdanauAttentionattention import BahdanauAttention # type: ignore
from LuongAttention import LuongAttention
import tensorflow as tf

def main():
    # BahdanauAttention
    # W_size = 100
    # print("Hidden State를 {0}차원으로 Mapping\n".format(W_size))

    # attention = BahdanauAttention(W_size)

    # enc_state = tf.random.uniform((1, 10, 512))
    # dec_state = tf.random.uniform((1, 512))

    # context_vector, attention_weights = attention(enc_state, dec_state)

    # LuongAttention
    emb_dim = 512
    attention = LuongAttention(emb_dim)

    enc_state = tf.random.uniform((1, 10, emb_dim))
    dec_state = tf.random.uniform((1, emb_dim))

    context_vector, attention_weights = attention(enc_state, dec_state)

if __name__ == "__main__":
    main()
