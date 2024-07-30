import re

class TextPreprocessor:
    def __init__(self, text):
        self.text = text
    
    def clean_text(self):
        reg = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]")
        self.text = reg.sub('', self.text)
        return self.text