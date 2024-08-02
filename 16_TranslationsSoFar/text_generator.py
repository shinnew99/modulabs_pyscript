import tensorflow as tf

class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_text(self, init_sentence="<start>", max_len=20):
        test_input = self.tokenizer.texts_to_sequence([init_sentence])
        test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)

        end_token = self.tokenizer.word_index["<end>"]

        while True:
            predict = self.model(test_tensor)
            predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1]

            test_tensor = tf.concat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)

            if predict_word.numpy()[0] == end_token:
                break
            if test_tensor.shape[1] >= max_len:
                break

            generated = ""
            for word_index in test_tensor[0].numpy():
                generated += self.tokenizer.index_word[word_index] + " " 

            return generated