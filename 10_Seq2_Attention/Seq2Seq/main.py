from encoder import Encoder
from decoder import Decoder
import tensorflow as tf

def main():
    vocab_size = 30000
    emb_size = 256
    lstm_size = 512
    batch_size = 1
    sample_seq_len = 3

    print("Vocab Size: {0}". format(vocab_size))
    print("Embedding Size: {0}".format(emb_size))
    print("LSTM Size: {0}".format(lstm_size))
    print("Batch Size: {0}".format(batch_size))
    print("Sampel Sequence Length: {0}\n".format(sample_seq_len))

    # Encoder
    encoder = Encoder(vocab_size, emb_size, lstm_size)
    sample_input = tf.zeros((batch_size, sample_seq_len))
    sample_output = encoder(sample_input)  # 컨텍스트 벡터로 사용할 인코더 LSTM의 최종 State값

    # Decoder
    decoder = Decoder(vocab_size, emb_size, lstm_size)
    dec_output = decoder(sample_input, sample_output)  # Decoder.call(x, context_v)을 호출


if __name__ == "__main__":
    main()
    