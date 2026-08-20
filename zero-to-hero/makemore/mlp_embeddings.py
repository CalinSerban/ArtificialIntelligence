"""
makemore Part 2 — Step 2: the embedding table
==============================================
C is a lookup table: shape (vocab=27, emb_dim). Row i = the learned vector
for character i. C[X] replaces every index in X with its vector.

Fill the TODOs, then run:  ../.venv/bin/python mlp_embeddings.py
"""
import torch
from mlp_dataset import build_dataset, stoi, itos, block_size   # reuse Step 1

words = open("names.txt").read().splitlines()
X, Y = build_dataset(words)
print("X shape:", tuple(X.shape), "\n")

VOCAB = 27
EMB_DIM = 2                      # start tiny (2) so we can literally plot it later
g = torch.Generator().manual_seed(2147483647)

# TODO a: create the embedding table C, shape (VOCAB, EMB_DIM), random values.
#   C = torch.randn((VOCAB, EMB_DIM), generator=g)
C = torch.randn((VOCAB, EMB_DIM), generator=g)

# --- a single lookup, to see what C[index] does -----------------------------
print("C[5] (the vector for char 'e'):", C[5].tolist() if C is not None else None)

# TODO b: embed the WHOLE dataset in one shot: emb = C[X]
#   X is (num, 3) integers -> emb becomes (num, 3, EMB_DIM):
#   each of the 3 context chars replaced by its EMB_DIM-vector.
emb = C[X] if C is not None else None

if emb is not None:
    print("\nemb = C[X] shape:", tuple(emb.shape), "  (num, block_size, emb_dim)")
    print("\nexample 0's context indices:", X[0].tolist(),
          "->", "".join(itos[i.item()] for i in X[0]))
    print("example 0 embedded (3 chars x 2 numbers):\n", emb[0])
    # sanity: emb[0][2] should equal C[ X[0][2] ]
    print("\nemb[0][2] == C[X[0][2]] ?",
          torch.allclose(emb[0][2], C[X[0][2]]))