import re

class puntuations:
    # 노이즈 유형(1) 문장부호: Hi, ny name is John.
    def pad_punctuation(sentence, punc):
        for p in punc:
            sentence = sentence.replace(p, " " + p + " ")

        return sentence


    def cleaning_text(text, punc, regex):
        # 노이즈 유형 (1) 문장부호 공백추가
        for p in punc:
            text = text.replace(p, " " + p + " ")

        # 노이즈 유형 (2), (3) 소문자화 및 특수문자 제거
        text = re.sub(regex, " ", text).lower()

        return text

