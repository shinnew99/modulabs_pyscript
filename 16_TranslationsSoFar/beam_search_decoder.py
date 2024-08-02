import math
import numpy as np

class BeamSearchDecoder:
    @staticmethod
    def decode(prob, beam_size):
        sequences = [[[], 1.0]]  # 생성된 문장과 점수를 저장

        for tok in prob:
            all_candidates = []

            for seq, score in sequences:
                for idx, p in enumerate(tok):  # 각 단어의 확률을 총점에 누적 곱
                    candidate = [seq + [idx], score * -math.log(-(p-1))]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)  # 총점 순 정렬
            sequences = ordered[:beam_size]  # Beam Size에 해당하는 문장만 저장 

        return sequences
