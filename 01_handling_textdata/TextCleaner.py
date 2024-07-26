import re

class TextCleaner:
    def __init__(self, punc, regex):
        self.punc = punc
        self.regex = regex

    def pad_punctuation(self, sentence):
        for p in self.punc:
            sentence = sentence.replace(p, " " + p + " ")

        return sentence

    def cleaning_text(self, text):
        # 노이즈 유형 (1) 문장부호 공백추가
        for p in self.punc:
            text = text.replace(p, " " + p + " ")

        # 노이즈 유형 (2), (3) 소문자화 및 특수문자 제거
        text = re.sub(self.regex, " ", text).lower()
        return text

# Usage example
if __name__ == "__main__":
    punc = [".", ",", "!", "?"]
    regex = r"[^a-zA-Z0-9\s]"

    cleaner = TextCleaner(punc, regex)

    sample_sentence = "Hi, my name is John."
    print(cleaner.pad_punctuation(sample_sentence))

    sample_text = "Hi, my name is John."
    print(cleaner.cleaning_text(sample_text))