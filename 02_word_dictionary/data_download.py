import requests
import tarfile
import os

# 파일 다운로드
url = 'https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.train.tar.gz'
response = requests.get(url, stream=True)

# 다운로드할 파일 경로 설정
download_path = r'C:\Users\신유진\workspaces\vscodeprojects\modulabnodes\aiffel\NLP\02_word_dictionary\sp_tokenizer\data\korean-english-park.train.tar.gz'

# 파일 저장
with open(download_path, 'wb') as f:
    f.write(response.raw.read())

# 파일 압축 해제
with tarfile.open(download_path, 'r:gz') as tar:
    tar.extractall(path=r'C:\Users\신유진\workspaces\vscodeprojects\modulabnodes\aiffel\NLP\02_word_dictionary\sp_tokenizer\data')

# 압축 해제 후 파일 경로
extracted_file_path = r'C:\Users\신유진\workspaces\vscodeprojects\modulabnodes\aiffel\NLP\02_word_dictionary\sp_tokenizer\data\korean-english-park.train.ko'
