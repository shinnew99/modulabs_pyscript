import os

import tensorflow as tf
import numpy as np

# from mecab import MeCab
from konlpy.tag import Mecab
import matplotlib.pyplot as plt

# %matplotlib inline

class data_download:

    def __init__(self):
        pass

    def process_text_file(file_path):
        mecab = Mecab()
        # print(mecab.morphs('Sample Text')) # 자연어처리가너무재밌어서밥먹는것도가끔까먹어요'

        with open(file_path, "r") as f:
            raw = f.read().splitlines()

        print("Data Size:", len(raw))
        print("Example:")
        print(raw[0])  # 예제로 첫 번째 줄 출력 (필요에 따라 수정 가능)
        
        # 중복 제거
        cleaned_corpus = list(set(raw))
        print("Cleaned Data Size:", len(cleaned_corpus))

        min_len = 999
        max_len = 0
        sum_len = 0

        for sen in cleaned_corpus:
            length = len(sen)
            if min_len > length:
                min_len = length
            if max_len < length:
                max_len = length
            sum_len += length

        print("문장의 최단 길이:", min_len)
        print("문장의 최장 길이:", max_len)
        print("문장의 평균 길이:", sum_len // len(cleaned_corpus))

        sentence_length = np.zeros((max_len), dtype=int)

        for sen in cleaned_corpus:  # 중복이 제거된 코퍼스 기준
            sentence_length[len(sen)-1] += 1

        plt.bar(range(max_len), sentence_length, width=1.0)
        plt.title("Sentence Length Distribution")
        plt.xlable("Sentence Length")
        plt.xlable("Frequency")
        plt.show()


 
