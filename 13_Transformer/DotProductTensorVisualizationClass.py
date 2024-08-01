import tensorflow as tf
import matplotlib.pyplot as plt

class DotProductTensorVisualization:
    def __init__(self, length, big_dim, small_dim):
        self.length = length
        self.big_dim = float(big_dim)
        self.small_dim = float(small_dim)
        self.big_tensor = self._make_dot_product_tensor((length, int(big_dim)))
        self.scaled_big_tensor = self.big_tensor / tf.sqrt(self.big_dim)
        self.small_tensor = self._make_dot_product_tensor((length, int(small_dim)))
        self.scaled_small_tensor = self.small_tensor / tf.sqrt(self.small_dim)
    
    def _make_dot_product_tensor(self, shape):
        A = tf.random.uniform(shape, minval=-3, maxval=3, dtype=tf.float32)
        B = tf.transpose(tf.random.uniform(shape, minval=-3, maxval=3, dtype=tf.float32), [1, 0])
        return tf.tensordot(A, B, axes=1)
    
    def plot_tensors(self):
        fig = plt.figure(figsize=(13, 6))

        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        ax1.set_title('1) Big Tensor')
        ax2.set_title('2) Big Tensor (Scaled)')
        ax3.set_title('3) Small Tensor')
        ax4.set_title('4) Small Tensor (Scaled)')

        ax1.imshow(tf.nn.softmax(self.big_tensor, axis=-1).numpy(), cmap='cividis')
        ax2.imshow(tf.nn.softmax(self.scaled_big_tensor, axis=-1).numpy(), cmap='cividis')
        ax3.imshow(tf.nn.softmax(self.small_tensor, axis=-1).numpy(), cmap='cividis')
        ax4.imshow(tf.nn.softmax(self.scaled_small_tensor, axis=-1).numpy(), cmap='cividis')

        plt.show()

# Usage example
length = 30
big_dim = 1024
small_dim = 10

dot_product_vis = DotProductTensorVisualization(length, big_dim, small_dim)
dot_product_vis.plot_tensors()