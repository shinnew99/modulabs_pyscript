from data_loader import DataLoader
from sentence_statistics import SentenceStatistics
from corpus_cleaner import CorpusCleaner
from tokenizer import Tokenizer
from mecab_tokenizer import MecabTokenizer

if __name__ == "__main__":
    path_to_file = r'C:\Users\[##UserName]\workspaces\vscodeprojects\modulabnodes\aiffel\NLP\02_word_dictionary\sp_tokenizer\data\korean-english-park.train.ko'
    # 경로는 본인 환경에 맞게!


    # Data Loading
    data_loader = DataLoader(path_to_file)
    raw_data = data_loader.get_data()
    data_loader.print_examples()

    # Sentence Statistics
    stats = SentenceStatistics(raw_data)
    stats.print_statistics()
    stats.plot_sentence_length_distribution()
    stats.check_sentence_with_length(1)
    stats.find_outliers()

    # Corpus Cleaning
    cleaner = CorpusCleaner(raw_data)
    cleaner.print_statistics()
    cleaned_corpus = cleaner.get_cleaned_corpus()
    filtered_corpus = cleaner.filter_corpus()
    stats_filtered = SentenceStatistics(filtered_corpus)
    stats_filtered.plot_sentence_length_distribution()

    # Tokenization
    tokenizer = Tokenizer()
    split_corpus = [kor.split() for kor in filtered_corpus]
    split_tensor, split_tokenizer = tokenizer.tokenize(split_corpus)
    tokenizer.print_vocabulary()

    # MeCab Tokenization
    mecab_tokenizer = MecabTokenizer()
    mecab_tensor, mecab_tokenizer = mecab_tokenizer.tokenize(filtered_corpus)
    mecab_tokenizer.print_vocabulary(mecab_tokenizer)

    # Tokenized Texts Examples
    print("Case 1")
    texts = mecab_tokenizer.tokenizer.sequences_to_texts([mecab_tensor[100]])
    print(texts[0])

    print("Case 2")
    sentence = " ".join(mecab_tokenizer.tokenizer.index_word[w] for w in mecab_tensor[100] if w != 0)
    print(sentence)