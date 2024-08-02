import sentencepiece as spm

def generate_tokenizer(corpus, vocab_size, lang="spa-eng", pad_id=0, bos_id=1, eos_id=2, unk_id=3):
    file = "./%s_corpus.txt" % lang
    model = "%s_spm" % lang

    with open(file, 'w') as f:
        for row in corpus: f.write(str(row) + '\n')

    spm.SentencePieceTrainer.Train(
        '--input=./%s --model_prefix=%s --vocab_size=%d --pad_id=%d --bos_id=%d --eos_id=%d --unk_id=%d' % (file, model, vocab_size, pad_id, bos_id, eos_id, unk_id)
    )

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('%s.model' % model)

    return tokenizer

def make_corpus(sentences, tokenizer):
    corpus = []
    for sentence in sentences:
        tokens = tokenizer.encode_as_ids(sentence)
        corpus.append(tokens)
    return corpus
