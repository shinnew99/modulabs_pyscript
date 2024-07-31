import tensorflow as tf

class Seq2SeqModel:
    def __init__(self, encoder, decoder, optimizer, loss_fn):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, src, tgt, dec_tok):
        bsz = src.shape[0]
        loss = 0

        with tf.GradientTape() as tape:
            enc_out = self.encoder(src)
            h_dec = enc_out[:, -1]
            dec_src = tf.expand_dims([dec_tok.word_index['<start>']] * bsz, 1)

            for t in range(1, tgt.shape[1]):
                pred, h_dec, _ = self.decoder(dec_src, h_dec, enc_out)
                loss += self.loss_fn(tgt[:, t], pred)
                dec_src = tf.expand_dims(tgt[:, t], 1)

        batch_loss = (loss / int(tgt.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss