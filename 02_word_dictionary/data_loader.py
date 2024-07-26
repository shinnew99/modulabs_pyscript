import os

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.raw_data = self._load_data()

    def _load_data(self):
        with open(self.path, "r") as f:
            raw = f.read().splitlines()
        return raw

    def get_data(self):
        return self.raw_data

    def print_examples(self, num_examples=5, step=20):
        print("Example:")
        for sen in self.raw_data[0:num_examples * step:step]:
            print(">>", sen)