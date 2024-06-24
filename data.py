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

        # Split the encoded data into training and validation
        self.encoded = self.encode(text)
        cut = int(len(self.encoded) * 0.9)
        self.train = self.encoded[:cut]
        self.val = self.encoded[cut:]

    def encode(self, s):
        return np.array([self.stoi[ch] for ch in s], dtype=np.uint16)

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])


def generate_add():
    lhs = random.randrange(1000)
    rhs = random.randrange(1000)
    return f"{lhs}+{rhs}={lhs+rhs}"


def generate_rev():
    n = str(random.randrange(1000000))
    rev_n = n[::-1]
    return f"R{n}={rev_n}"


def save_data(data):
    with open(data_path, "w") as f:
        f.write("\n".join(data + []))


# Change this to change what everything does
def generate_one():
    return generate_add()


def generate():
    "Generates data and returns an Encoding for it."
    random.seed(1337)
    data = [generate_one() for _ in range(100000)]
    save_data(data)
    return Encoding()


if __name__ == "__main__":
    generate()
    print("done generating data.")
