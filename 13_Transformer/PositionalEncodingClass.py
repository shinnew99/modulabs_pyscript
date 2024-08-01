import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding:
    def __init__(self, pos, d_model):
        self.pos = pos
        self.d_model = d_model
        self.positional_encoding = self._calculate_positional_encoding()
    
    def _calculate_positional_encoding(self):
        def cal_angle(position, i):
            return position / np.power(10000, int(i) / self.d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, i) for i in range(self.d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(self.pos)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return sinusoid_table
    
    def get_positional_encoding(self):
        return self.positional_encoding

    def print_positional_encoding(self):
        print("Positional Encoding 값:\n", self.positional_encoding)
    
    def plot_positional_encoding(self):
        plt.figure(figsize=(7, 7))
        plt.imshow(self.positional_encoding, cmap='Blues')
        plt.title('Positional Encoding')
        plt.xlabel('Depth')
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()

# Usage example
pos = 7
d_model = 4

pos_enc = PositionalEncoding(pos, d_model)
pos_enc.print_positional_encoding()

# Additional specific value checks
i = 0
print("")
print("if pos == 0, i == 0: ", np.sin(0 / np.power(10000, 2 * i / d_model)))
print("if pos == 1, i == 0: ", np.sin(1 / np.power(10000, 2 * i / d_model)))
print("if pos == 2, i == 0: ", np.sin(2 / np.power(10000, 2 * i / d_model)))
print("if pos == 3, i == 0: ", np.sin(3 / np.power(10000, 2 * i / d_model)))

print("")
print("if pos == 0, i == 1: ", np.cos(0 / np.power(10000, 2 * i + 1 / d_model)))
print("if pos == 1, i == 1: ", np.cos(1 / np.power(10000, 2 * i + 1 / d_model)))
print("if pos == 2, i == 1: ", np.cos(2 / np.power(10000, 2 * i + 1 / d_model)))
print("if pos == 3, i == 1: ", np.cos(3 / np.power(10000, 2 * i + 1 / d_model)))

# Plot the positional encoding
pos_enc.plot_positional_encoding()