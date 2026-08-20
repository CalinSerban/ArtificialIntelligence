"""
makemore Part 2 — Step 3: flatten embeddings + hidden layer
=============================================================
So far: emb = C[X], shape (num, block_size=3, emb_dim=2).
A linear layer needs 2D input: (num, features). So we flatten the last
two dims together: (num, 3, 2) -> (num, 6).

Then: hidden layer, same shape as micrograd/bigram neurons -- multiply by
weights, add bias, squash with tanh.
    h = tanh(emb_flat @ W1 + b1)

Fill the TODOs, then run:  ../.venv/bin/python mlp_hidden.py
"""
import torch
from mlp_dataset import build_dataset, stoi, itos, block_size

words = open("names.txt").read().splitlines()
X, Y = build_dataset(words)

VOCAB = 27
EMB_DIM = 2
N_HIDDEN = 100          # size of the hidden layer -- a free choice, tune later

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((VOCAB, EMB_DIM), generator=g)
emb = C[X]
print("emb shape:", tuple(emb.shape))

# TODO a: flatten emb from (num, block_size, EMB_DIM) to (num, block_size*EMB_DIM).
#   emb_flat = emb.view(emb.shape[0], -1)
emb_flat = emb.view(emb.shape[0], -1)
print("emb_flat shape:", tuple(emb_flat.shape) if emb_flat is not None else None,
      " (should be (num, 6))")

# TODO b: create W1, b1 for the hidden layer.
#   input dim = block_size*EMB_DIM (6), output dim = N_HIDDEN
#   W1 = torch.randn((block_size*EMB_DIM, N_HIDDEN), generator=g)
#   b1 = torch.randn(N_HIDDEN, generator=g)
W1 = torch.randn((block_size*EMB_DIM, N_HIDDEN), generator=g)
b1 = torch.randn(N_HIDDEN, generator=g)

# TODO c: compute the hidden layer activations.
#   h = torch.tanh(emb_flat @ W1 + b1)
h = torch.tanh(emb_flat @ W1 + b1)

if h is not None:
    print("h shape:", tuple(h.shape), " (should be (num, N_HIDDEN))")
    print("h[0][:10]:", h[0][:10].tolist(), " (should all be in [-1, 1], tanh's range)")
