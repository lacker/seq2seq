import numpy as np
import os
import random

data_path = os.path.join(os.path.dirname(__file__), "data.txt")


class Encoding:
    def __init__(self):
        print("creating encoding...")
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

    def make_decoder_only_datasets(self, window_size):
        """
        Returns ((train_inputs, train_outputs), (val_inputs, val_outputs)) datasets.
        Inputs have shape (num datapoints, window size).
        outputs have shape (num datapoints, window size).
        """
        # Split the encoded data into training and validation
        cut = int(len(self.lines) * 0.9)
        train = self.make_decoder_only_dataset(self.lines[:cut], window_size)
        val = self.make_decoder_only_dataset(self.lines[cut:], window_size)
        return train, val

    def make_decoder_only_dataset(self, lines, window_size):
        """
        Returns (inputs, outputs) for decoder-only.
        Inputs have shape (num datapoints, window size).
        outputs have shape (num datapoints, window size).
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
        return np.array(inputs, dtype=np.uint16), np.array(outputs, dtype=np.uint16)

    def make_ed_datasets(self, window_size):
        """
        Returns a tuple of (in, out, target) for both train and val.
        """
        # Split the encoded data into training and validation
        cut = int(len(self.lines) * 0.9)
        train = self.make_ed_dataset(self.lines[:cut], window_size)
        val = self.make_ed_dataset(self.lines[cut:], window_size)
        return train, val

    def make_ed_dataset(self, lines, window_size):
        """
        Returns (input tokens, output tokens, targets) for encoder-decoder.
        All have shape (num datapoints, window size).
        """
        inputs = []
        outputs = []
        targets = []
        for line in lines:
            question, answer = line.split("=")
            right_pad_size = window_size - len(question)
            assert right_pad_size >= 0
            encoded_question = self.encode(question + ("." * right_pad_size))

            # Need to pad the answer one more to account for both outputs and targets
            right_pad_size = window_size - len(answer) + 1
            assert right_pad_size >= 1
            encoded_answer = self.encode(answer + ("." * right_pad_size))

            inputs.append(encoded_question)
            outputs.append(encoded_answer[:-1])
            targets.append(encoded_answer[1:])
        return inputs, outputs, targets

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


def generate():
    "Generates data and returns an Encoding for it."
    random.seed(1337)
    data = [generate_one() for _ in range(100000)]
    save_data(data)
    return Encoding()


if __name__ == "__main__":
    generate()
    print("done generating data.")
