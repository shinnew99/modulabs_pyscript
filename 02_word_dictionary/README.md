02_word_dictionary/
│
├── data/
│   ├── korean-english-park.train.ko
│   ├── korean-english-park.train.tar.gz
│   └── korean-english-park.train.tar.gz
│
├
│── data_loader.py
│── sentence_statistics.py
│── corpus_cleaner.py
│── tokenizer.py
│── mecab_tokenizer.py
├── main.py
└── README.md

I might erase 'data' if space not available.

# data_loader.py
- get_data()
- print_examples()

# sentence_statistics.py
- _calculate_lengths()
- print_statistics()
- _calculate_sentence_length()
- plot_sentence_length_distribution()
- check_sentence_with_length()
- find_outliers()

# corpus_cleaner.py
- _remove_duplicates()
- get_cleaned_corpus()
- filter_corpus()
- print_statistics()

# tokenizer.py
- tokenize()
- print_vocabulary()

# mecab_tokenizer()
- mecab_split()
- tokenize()
- print_vocabulary()

# main()
- 모든 예제 여기로 들어감, 출력 후 결과물 figure.1png
