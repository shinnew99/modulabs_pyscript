import numpy as np
from beam_search_decoder import BeamSearchDecoder
from vocab import Vocabulary

# prob_seq은 문장의 각 위치에서 어떤 단어가 생성될지의 확률을 한 번에 정의해둔 것입니다.
# 실제로는 각 단어에 대한 확률이 prob_seq처럼 한번에 정의되지 않기 때문에 실제 문장 생성과정과는 거리가 멉니다.
# 하지만 Beam Search의 동작과정 이해를 돕기위해 이와 같은 예시를 보여드립니다.
# prob_seq의 각 열은 위 vocab의 각 숫자(key)에 대응됩니다.
prob_seq = [[0.01, 0.01, 0.60, 0.32, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # 커피 : 0.60
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.75, 0.01, 0.01, 0.17],  # 를 : 0.75
            [0.01, 0.01, 0.01, 0.35, 0.48, 0.10, 0.01, 0.01, 0.01, 0.01],  # 가져 : 0.48
            [0.24, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.68],  # 도 : 0.68
            [0.01, 0.01, 0.12, 0.01, 0.01, 0.80, 0.01, 0.01, 0.01, 0.01],  # 될 : 0.80
            [0.01, 0.81, 0.01, 0.01, 0.01, 0.01, 0.11, 0.01, 0.01, 0.01],  # 까요? : 0.81
            [0.70, 0.22, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # <pad> : 0.91
            [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # <pad> : 0.91
            [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # <pad> : 0.91
            [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]  # <pad> : 0.91


prob_seq = np.array(prob_seq)
beam_size = 3

result = BeamSearchDecoder.decode(prob_seq, beam_size)

for seq, score in result:
    sentence = ""

    for word in seq:
        sentence += Vocabulary.vocab[word] + " "

    print(sentence, "// Score: %.4f" % score)


# Q. beam_size 인자 값을 바꿔보세요.
beam_size = 5  # Change this value to experiment with different beam sizes

result = BeamSearchDecoder.decode(prob_seq, beam_size)

for seq, score in result:
    sentence = ""

    for word in seq:
        sentence += Vocabulary.vocab[word] + " "

    print(sentence, "// Score: %.4f" % score)