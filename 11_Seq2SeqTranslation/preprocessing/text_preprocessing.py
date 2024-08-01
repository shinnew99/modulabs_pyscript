import re

class TextPreprocessing:
    @staticmethod
    def preprocess_sentence(sentence, s_token=False, e_token=False):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()

        if s_token:
            sentence = '<start> ' + sentence

        if e_token:
            sentence += ' <end>'
        
        return sentence