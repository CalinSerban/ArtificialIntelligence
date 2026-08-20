"""
makemore Part 2 — Step 1: build the context-window dataset
==========================================================
Bigram used 1 previous char. Now we use the previous `block_size` chars
(a "context") to predict the next one.

Example for the name "emma" with block_size=3 (pad the start with '.'):
    context      -> target
    . . .        -> e
    . . e        -> m
    . e m        -> m
    e m m        -> a
    m m a        -> .

So each training example X is a list of 3 char-indices, and Y is the next
char-index.

Fill the TODO, then run:  ../.venv/bin/python mlp_dataset.py
"""
import torch

words = open("names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

block_size = 3   # how many previous chars we condition on

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size          # start padded with '.' (index 0)
        for ch in w + ".":                  # each real char, then the end token
            ix = stoi[ch]
            # TODO a: append the CURRENT context (a copy) to X, and ix to Y.
            #   X.append(context); Y.append(ix)
            X.append(context.copy())
            Y.append(ix)
            # TODO b: slide the context window forward by one:
            #   drop the oldest char, add ix at the end:
            #   context = context[1:] + [ix]
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

X, Y = build_dataset(words)
print("X shape:", tuple(X.shape), "  Y shape:", tuple(Y.shape))
print("(X should be (num_examples, 3);  Y should be (num_examples,))\n")

# show the first few examples decoded, to eyeball correctness
print("first 8 examples (context -> target):")
for i in range(min(8, len(X))):
    ctx = "".join(itos[ix.item()] for ix in X[i])
    print(f"  {ctx} -> {itos[Y[i].item()]}")
