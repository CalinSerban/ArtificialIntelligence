"""
makemore Part 5 — Step 2: dataset at block_size=8
=====================================================
Same build_dataset logic as mlp_dataset.py -- sliding-window context ->
next-char pairs -- just parameterized by block_size instead of hardcoded
to 3. WaveNet needs a longer context (8) so FlattenConsecutive actually has
something to hierarchically fuse (8 -> 4 -> 2 -> 1 needs a power of 2).

Nothing new to fill in here -- you already wrote this logic once in
mlp_dataset.py. Run directly to sanity-check the shapes:
  ../.venv/bin/python wavenet_dataset.py
"""
import torch
from mlp_dataset import stoi, itos

block_size = 8

def build_dataset(words, block_size=block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

if __name__ == "__main__":
    words = open("names.txt").read().splitlines()
    X, Y = build_dataset(words)
    print("X shape:", tuple(X.shape), " Y shape:", tuple(Y.shape))
    print("(X should be (num_examples, 8);  Y should be (num_examples,))\n")
    print("first 3 examples (context -> target):")
    for i in range(3):
        ctx = "".join(itos[ix.item()] for ix in X[i])
        print(f"  {ctx!r} -> {itos[Y[i].item()]}")
