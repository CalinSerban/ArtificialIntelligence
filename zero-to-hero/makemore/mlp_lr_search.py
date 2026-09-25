"""
makemore Part 2 — Step 6: the learning-rate range test
=========================================================
So far LR=0.1 was a guess. Karpathy's trick for picking a real one: sweep
LR exponentially from tiny to huge over a run, plot loss against log10(lr),
and read off where the curve is dropping fastest right before it blows up.

How the sweep works: instead of a fixed LR, on step `i` we use `lrs[i]`,
where `lrs` covers a WIDE exponential range (e.g. 10^-3 up to 10^0). Linear
spacing would waste almost all its steps on huge values (0.001 to 1.0 is a
1000x range -- linear steps barely sample the small end). Exponential
(evenly spaced in log10-space) samples that range uniformly on a scale
that actually matches how sensitive training is to LR.

Fill the TODOs, then run:  ../.venv/bin/python mlp_lr_search.py
Output: lr_search.png -- loss vs log10(lr).
"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlp_dataset import build_dataset, stoi, itos, block_size

words = open("names.txt").read().splitlines()

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

VOCAB = 27
EMB_DIM = 10
N_HIDDEN = 200
BATCH_SIZE = 32
STEPS = 1000   # short run -- we're exploring, not training to convergence

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((VOCAB, EMB_DIM), generator=g)
W1 = torch.randn((block_size * EMB_DIM, N_HIDDEN), generator=g)
b1 = torch.randn(N_HIDDEN, generator=g)
W2 = torch.randn((N_HIDDEN, VOCAB), generator=g) * 0.01
b2 = torch.randn(VOCAB, generator=g) * 0.0
params = [C, W1, b1, W2, b2]
for p in params:
    p.requires_grad = True

# TODO a: build the exponential LR range. We want STEPS values, evenly
# spaced in EXPONENT space from 10^-3 to 10^0 (0.001 to 1.0).
#   lre = torch.linspace(-3, 0, STEPS)   # the exponents, evenly spaced
#   lrs = 10 ** lre                      # the actual LR values
lre = torch.linspace(-3, 0, STEPS)
lrs = 10 ** lre

print(f"LR range test: {lrs[0]:.5f} to {lrs[-1]:.5f} over {STEPS} steps")

lri = []     # log10(lr) used at each step
lossi = []   # loss at each step

for step in range(STEPS):
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)

    emb = C[Xtr[ix]]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in params:
        p.grad = None
    loss.backward()

    # TODO b: use this step's LR from the sweep, instead of a fixed constant.
    #   lr = lrs[step]
    lr = lrs[step]

    for p in params:
        p.data -= lr * p.grad

    # TODO c: record what we need for the plot: this step's exponent
    # (lre[step], as a plain float) and this step's loss.
    #   lri.append(lre[step].item())
    #   lossi.append(loss.item())
    lri.append(lre[step].item())
    lossi.append(loss.item())

    if step % 100 == 0 or step == STEPS - 1:
        print(f"  step {step:4d}  lr {lr:.5f}  loss {loss.item():.4f}")

plt.figure(figsize=(8, 5))
plt.plot(lri, lossi)
plt.xlabel("log10(learning rate)")
plt.ylabel("loss")
plt.title("LR range test -- pick the LR where this is dropping fastest,\n"
          "just before it turns upward / explodes")
plt.savefig("lr_search.png")
print("saved lr_search.png")
