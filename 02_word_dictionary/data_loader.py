import os


# 기존 아이펠 경로를 그대로 사용할때는 이 코드 사용하면 됨,
# class DataLoader:
#     def __init__(self, path):
#         self.path = path
#         self.raw_data = self._load_data()

#     def _load_data(self):
#         with open(self.path, "r") as f:
#             raw = f.read().splitlines()
#         return raw

#     def get_data(self):
#         return self.raw_data

#     def print_examples(self, num_examples=5, step=20):
#         print("Example:")
#         for sen in self.raw_data[0:num_examples * step:step]:
#             print(">>", sen)

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw = f.read().splitlines()
        return raw

    def print_examples(self, num_examples=5):
        raw_data = self.get_data()
        print("Data Size:", len(raw_data))
        print("Example:")
        for sen in raw_data[:num_examples]:
            print(">>", sen)