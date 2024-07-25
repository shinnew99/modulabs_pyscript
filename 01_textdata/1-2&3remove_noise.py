import re


# 노이즈 유형(1) 문장부호: Hi, ny name is John.
def pad_punctuation(sentence, punc):
    for p in punc:
        sentence = sentence.replace(p, " " + p + " ")

    return sentence

sentence1 = "Hi, my name is john."
print(pad_punctuation(sentence1, [".", "?", "!", ","]))


# 노이즈 유형(2) 대소 문자: First, open the first chapter.
sentence2 = "First, open the first chapter."
# print(sentence2.lower())


# Q. sentence의 모든 단어를 대문자로 바꿔보세요. 
# 힌트: upper() 함수를 사용해 보세요!
print(sentence1.upper()) # sentence2도 됨



# 노이즈 유형(3) 특수문자: He is a 10-year old boy.
sentence3 = "He is a ten-year-old boy."
sentence3 = re.sub("([^a-zA-Z.,?!])", " ", sentence3)
print(sentence3)





# From The Project Gutenberg
# (https://www.gutenberg.org/files/2397/2397-h/2397-h.htm)

corpus = """
In the days that followed I learned to spell in this uncomprehending way a great many words, among them pin, hat, cup and a few verbs like sit, stand and walk. 
But my teacher had been with me several weeks before I understood that everything has a name.
One day, we walked down the path to the well-house, attracted by the fragrance of the honeysuckle with which it was covered. 
Some one was drawing water and my teacher placed my hand under the spout. 
As the cool stream gushed over one hand she spelled into the other the word water, first slowly, then rapidly. 
I stood still, my whole attention fixed upon the motions of her fingers. 
Suddenly I felt a misty consciousness as of something forgotten—a thrill of returning thought; and somehow the mystery of language was revealed to me. 
I knew then that "w-a-t-e-r" meant the wonderful cool something that was flowing over my hand. 
That living word awakened my soul, gave it light, hope, joy, set it free! 
There were barriers still, it is true, but barriers that could in time be swept away.""" 

def cleaning_text(text, punc, regex):
    # 노이즈 유형 (1) 문장부호 공백추가
    for p in punc:
        text = text.replace(p, " " + p + " ")

    # 노이즈 유형 (2), (3) 소문자화 및 특수문자 제거
    text = re.sub(regex, " ", text).lower()

    return text

print(cleaning_text(corpus, [".", ",", "!", "?"], "([^a-zA-Z0-9.,?!\n])"))
