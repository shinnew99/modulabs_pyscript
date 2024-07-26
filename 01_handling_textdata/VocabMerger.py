import re
import collections

class VocabMerger:
    def __init__(self, vocab, num_merges=5):
        self.vocab = vocab
        self.num_merges = num_merges
        self.token_vocab = []

    def get_stats(self):
        """
        단어 사전을 불러와 단어는 공백 단위로 쪼개어 문자 list를 만들고 빈도수와 쌍을 이루게 합니다. (symbols)
        """
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair):
        """
        문자 쌍(pair)과 단어 리스트(v_in)를 입력받아 각각의 단어에서 등장하는 문자 쌍을 치환합니다. (하나의 글자처럼 취급)
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in self.vocab:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = self.vocab[word]

        return v_out, pair[0] + pair[1]

    def perform_merges(self):
        for i in range(self.num_merges):
            print(f">> Step {i + 1}")

            pairs = self.get_stats()
            best = max(pairs, key=pairs.get)  # 가장 많은 빈도수를 가진 문자 쌍을 반환합니다.
            self.vocab, merge_tok = self.merge_vocab(best)
            print("다음 문자 쌍을 치환:", merge_tok)
            print("변환된 Vocab:\n", self.vocab, "\n")

            self.token_vocab.append(merge_tok)

        print("Merged Vocab:", self.token_vocab)

# Usage example
if __name__ == "__main__":
    vocab = {
        'l o w ': 5,
        'l o w e r ': 2,
        'n e w e s t ': 6,
        'w i d e s t ': 3
    }

    merger = VocabMerger(vocab)
    merger.perform_merges()