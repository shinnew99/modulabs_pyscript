import numpy as np
import matplotlib.pyplot as plt

class SentenceStatistics:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.min_len, self.max_len, self.sum_len = self._calculate_lengths()
        self.sentence_length = self._calculate_sentence_length()

    def _calculate_lengths(self):
        min_len = 999
        max_len = 0
        sum_len = 0
        for sen in self.raw_data:
            length = len(sen)
            if min_len > length: min_len = length
            if max_len < length: max_len = length
            sum_len += length
        return min_len, max_len, sum_len

    def print_statistics(self):
        print("문장의 최단 길이:", self.min_len)
        print("문장의 최장 길이:", self.max_len)
        print("문장의 평균 길이:", self.sum_len // len(self.raw_data))

    def _calculate_sentence_length(self):
        sentence_length = np.zeros((self.max_len), dtype=int)
        for sen in self.raw_data:
            sentence_length[len(sen) - 1] += 1
        return sentence_length

    def plot_sentence_length_distribution(self):
        plt.bar(range(self.max_len), self.sentence_length, width=1.0)
        plt.title("Sentence Length Distribution")
        plt.show()

    def check_sentence_with_length(self, length, limit=100):
        count = 0
        for sen in self.raw_data:
            if len(sen) == length:
                print(sen)
                count += 1
                if count >= limit: return

    def find_outliers(self, threshold=1500):
        for idx, _sum in enumerate(self.sentence_length):
            if _sum > threshold:
                print("Outlier Index:", idx + 1)