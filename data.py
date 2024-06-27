from dataclasses import dataclass
import numpy as np
import os
import random
import torch

data_path = os.path.join(os.path.dirname(__file__), "data.txt")


@dataclass
class Batch:
    inputs: torch.Tensor
    outputs: torch.Tensor


@dataclass
class DataSubset:
    # A subset is typically "train" or "val".
    # Both inputs and outputs have shape (num datapoints, window size).
    inputs: np.ndarray
    outputs: np.ndarray


class Dataset:
    def __init__(self, window_size, training=False):
        print("loading dataset...")
        self.window_size = window_size
        self.stoi = {".": 0}
        self.itos = {0: "."}
        self.lines = [line.strip() for line in open(data_path)]
        chars = set()
        for line in self.lines:
            for ch in line:
                assert ch != ".", "data should not contain dots"
                chars.add(ch)
        for ch in sorted(chars):
            self.stoi[ch] = len(self.stoi)
            self.itos[self.stoi[ch]] = ch
            print(f"{repr(ch)} -> {self.stoi[ch]}")
        self.vocab_size = len(self.stoi)

        if training:
            # Split the encoded data into training and validation
            cut = int(len(self.lines) * 0.9)
            self.train = self.make_subset(self.lines[:cut], window_size)
            self.val = self.make_subset(self.lines[cut:], window_size)

    def make_subset(self, lines, window_size):
        """
        Constructs a DataSubset from the provided lines.
        """
        inputs = []
        outputs = []
        for line in lines:
            question, answer = line.split("=")
            # We need enough left-padding to make a window whose last character is the =
            left_pad_size = window_size - len(question) - 1
            assert left_pad_size >= 0
            padded = ("." * left_pad_size) + line + "."
            encoded_padded = self.encode(padded)
            for i in range(len(encoded_padded) - window_size):
                inputs.append(encoded_padded[i : i + window_size])
                outputs.append(encoded_padded[i + 1 : i + window_size + 1])
        np_inputs = np.array(inputs, dtype=np.uint16)
        np_outputs = np.array(outputs, dtype=np.uint16)
        return DataSubset(inputs=np_inputs, outputs=np_outputs)

    def encode(self, s):
        "Return a one-dimensional array."
        return np.array([self.stoi[ch] for ch in s], dtype=np.uint16)

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])


def generate_add():
    n = 3
    left = random.randrange(10**n)
    left_str = str(left).zfill(n)
    right = random.randrange(10**n)
    right_str = str(right).zfill(n)
    answer = left + right
    answer_str = str(answer).zfill(n + 1)
    return f"{left_str}+{right_str}={answer_str}"


def generate_mul():
    m = 2
    n = 4
    left = random.randrange(10**m)
    left_str = str(left).zfill(m)
    right = random.randrange(10**n)
    right_str = str(right).zfill(n)
    answer = left * right
    answer_str = "".join(str(answer).zfill(m + n))
    return f"{left_str}*{right_str}={answer_str}"


def generate_rev():
    n = str(random.randrange(1000000))
    rev_n = n[::-1]
    return f"R{n}={rev_n}"


def save_data(data):
    with open(data_path, "w") as f:
        f.write("\n".join(data + []))


# Change this to change what everything does
def generate_one():
    return generate_mul()


def generate_dataset(window_size):
    "Generates data and returns a Dataset for it."
    random.seed(1337)
    data = [generate_one() for _ in range(100000)]
    save_data(data)
    return Dataset(window_size, training=True)


if __name__ == "__main__":
    generate_dataset()
    print("done generating data.")
