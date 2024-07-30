from attention import BahdanauAttention # type: ignore
import tensorflow as tf

def main():
    W_size = 100
    print("Hidden State를 {0}차원으로 Mapping\n".format(W_size))

    attention = BahdanauAttention(W_size)

    enc_state = tf.random.uniform((1, 10, 512))
    dec_state = tf.random.uniform((1, 512))

    context_vector, attention_weights = attention(enc_state, dec_state)

if __name__ == "__main__":
    main()
