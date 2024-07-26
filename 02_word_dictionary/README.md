02_word_dictionary/ <br>
│ <br>
├── data/ <br>
│   ├── korean-english-park.train.ko <br>
│   ├── korean-english-park.train.tar.gz <br>
│   └── korean-english-park.train.tar.gz <br>
│<br>
├<br>
│── data_loader.py <br>
│── sentence_statistics.py <br>
│── corpus_cleaner.py <br>
│── tokenizer.py <br>
│── mecab_tokenizer.py <br>
├── main.py <br>
└── README.md <br>
<br>
I might erase 'data' if space not available. <br>

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
