import re

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.strip()
    return sentence

def split_spa_eng_sentences(spa_eng_sentences):
    spa_sentences = []
    eng_sentences = []
    for spa_eng_sentence in spa_eng_sentences:
        eng_sentence, spa_sentence = spa_eng_sentence.split('\t')
        spa_sentences.append(spa_sentence)
        eng_sentences.append(eng_sentence)
    return eng_sentences, spa_sentences
