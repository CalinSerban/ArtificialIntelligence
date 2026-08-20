"""
makemore Part 2 — Step 4: output layer + loss
================================================
h is (num, N_HIDDEN). Add one more linear layer (no activation this time --
raw scores) to get (num, 27) logits, one score per possible next-char.
Then compute loss with F.cross_entropy -- same math as your manual
softmax+NLL from Part 1 (exp -> normalize -> pick -> -log -> mean), just
fused into one numerically-stable call.

Fill the TODOs, then run:  ../.venv/bin/python mlp_loss.py
"""
import torch
import torch.nn.functional as F
from mlp_dataset import build_dataset, stoi, itos, block_size

words = open("names.txt").read().splitlines()
X, Y = build_dataset(words)

VOCAB = 27
EMB_DIM = 2
N_HIDDEN = 100

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((VOCAB, EMB_DIM), generator=g)
emb = C[X]
emb_flat = emb.view(emb.shape[0], -1)
W1 = torch.randn((block_size * EMB_DIM, N_HIDDEN), generator=g)
b1 = torch.randn(N_HIDDEN, generator=g)
h = torch.tanh(emb_flat @ W1 + b1)
print("h shape:", tuple(h.shape))

# TODO a: create W2, b2 for the output layer.
#   input dim = N_HIDDEN, output dim = VOCAB (27)
#   W2 = torch.randn((N_HIDDEN, VOCAB), generator=g)
#   b2 = torch.randn(VOCAB, generator=g)
W2 = torch.randn((N_HIDDEN, VOCAB), generator=g)
b2 = torch.randn(VOCAB, generator=g)

# TODO d: unscaled randn init makes logits huge (confidently WRONG at
# random -> loss way above the ~3.3 uniform baseline). Shrink W2 and b2
# so the network starts close to "honestly uncertain" instead.
#   W2 = W2 * 0.01
#   b2 = b2 * 0.0
W2 *= 0.01
b2 *= 0.0

# TODO b: compute logits (no activation -- raw scores).
#   logits = h @ W2 + b2
logits = h @ W2 + b2

if logits is not None:
    print("logits shape:", tuple(logits.shape), " (should be (num, 27))")

# TODO c: compute the loss with F.cross_entropy(logits, Y)
loss = F.cross_entropy(logits, Y)

if loss is not None:
    print("loss:", loss.item(), " (random init -> should be roughly -log(1/27) ~= 3.3)")
