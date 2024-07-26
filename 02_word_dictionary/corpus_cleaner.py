class CorpusCleaner:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.cleaned_corpus = self._remove_duplicates()

    def _remove_duplicates(self):
        return list(set(self.raw_data))

    def get_cleaned_corpus(self):
        return self.cleaned_corpus

    def filter_corpus(self, min_len=10, max_len=150):
        return [s for s in self.cleaned_corpus if min_len <= len(s) < max_len]

    def print_statistics(self):
        min_len, max_len, sum_len = 999, 0, 0
        for sen in self.cleaned_corpus:
            length = len(sen)
            if min_len > length: min_len = length
            if max_len < length: max_len = length
            sum_len += length

        print("문장의 최단 길이:", min_len)
        print("문장의 최장 길이:", max_len)
        print("문장의 평균 길이:", sum_len // len(self.cleaned_corpus))