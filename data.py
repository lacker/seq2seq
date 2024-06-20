import numpy as np
import os
import random

data_path = os.path.join(os.path.dirname(__file__), "data.txt")


class Encoding:
    def __init__(self):
        print("creating encoding...")
        self.stoi = {}
        self.itos = {}
        text = open(data_path).read()
        chars = set()
        for ch in text:
            chars.add(ch)
        for ch in sorted(chars):
            self.stoi[ch] = len(self.stoi)
            self.itos[self.stoi[ch]] = ch
            print(f"{repr(ch)} -> {self.stoi[ch]}")
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return np.array([self.stoi[ch] for ch in s], dtype=np.uint8)

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])


def generate_add():
    lhs = random.randrange(1000)
    rhs = random.randrange(1000)
    return f"{lhs}+{rhs}={lhs+rhs}"


def save_data(data):
    with open(data_path, "w") as f:
        f.write("\n".join(data + []))


def generate_data():
    data = [generate_add() for _ in range(100000)]
    save_data(data)
    return data


if __name__ == "__main__":
    generate_data()
    print("done generating data.")
