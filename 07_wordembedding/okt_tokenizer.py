from konlpy.tag import Okt

class OktTokenizer:
    def __init__(self):
        self.okt = Okt()
    
    def tokenizer(self, text):
        tokens = self.okt.morphts(text)
        return tokens