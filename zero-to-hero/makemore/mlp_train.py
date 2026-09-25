"""
makemore Part 2 — Step 5: the real training loop
=====================================================
Same architecture as mlp_loss.py (embed -> flatten -> tanh hidden -> logits
-> cross_entropy), but now we actually TRAIN it: many steps, each on a
random MINIBATCH instead of the whole dataset.

Why minibatches? X has tens of thousands of rows. Computing the forward
pass over ALL of them every single step is slow. Instead, each step we
grab a random handful (BATCH_SIZE) of rows, compute loss+gradients on
just that handful, and update. It's a noisier estimate of the "true"
gradient (over the full dataset), but each step is way cheaper, so you
get far more updates per second of compute -- a good trade in practice.

Fill the TODOs, then run:  ../.venv/bin/python mlp_train.py
"""
import torch
import torch.nn.functional as F
from mlp_dataset import build_dataset, stoi, itos, block_size

words = open("names.txt").read().splitlines()

# TODO split: shuffle the WORDS (not individual examples -- so all of a
# given name's context/target pairs stay together in the same split, no
# leakage), then carve into 80% train / 10% dev / 10% test by index.
#   import random
#   random.seed(42)
#   random.shuffle(words)
#   n1 = int(0.8 * len(words))
#   n2 = int(0.9 * len(words))
#   Xtr,  Ytr  = build_dataset(words[:n1])
#   Xdev, Ydev = build_dataset(words[n1:n2])
#   Xte,  Yte  = build_dataset(words[n2:])
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print("train:", Xtr.shape[0], " dev:", Xdev.shape[0], " test:", Xte.shape[0])

VOCAB = 27
EMB_DIM = 10
N_HIDDEN = 200
BATCH_SIZE = 32
STEPS = 20000
LR = 0.1

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((VOCAB, EMB_DIM), generator=g)
W1 = torch.randn((block_size * EMB_DIM, N_HIDDEN), generator=g)
b1 = torch.randn(N_HIDDEN, generator=g)
W2 = torch.randn((N_HIDDEN, VOCAB), generator=g) * 0.01
b2 = torch.randn(VOCAB, generator=g) * 0.0
params = [C, W1, b1, W2, b2]

# TODO a: every parameter needs requires_grad=True for autograd to track it
# (unlike xor_nn.py, we didn't set it at creation time above -- set it now
# on each tensor in `params`).
#   for p in params: p.requires_grad = True
for p in params:
    p.requires_grad = True

lossi = []   # track loss over time so we can eyeball the trend at the end

for step in range(STEPS):
    # sample a random minibatch of BATCH_SIZE row-indices from the TRAIN split only
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)

    # forward pass, using only the sampled rows Xtr[ix], Ytr[ix]
    emb = C[Xtr[ix]]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in params:
        p.grad = None
    loss.backward()

    # TODO decay: LR=0.1 the whole way overshoots once the model gets close
    # to a good solution (that's the bouncing-around-instead-of-settling
    # you saw in the last run). Use a bigger LR early, a smaller one late.
    #   lr = 0.1 if step < STEPS * 0.75 else 0.01
    lr = LR if step < STEPS * 0.75 else 0.01

    for p in params:
        p.data -= lr * p.grad

    lossi.append(loss.item())
    if step % 2000 == 0 or step == STEPS - 1:
        print(f"  step {step:6d}  minibatch loss {loss.item():.4f}")
        print(f"ix shape: {ix.shape}")

# the number above is just ONE minibatch's loss -- noisy. For a fair read,
# evaluate on each split's FULL set of examples, no_grad since we're not
# training here. Same forward-pass code you've already written several
# times now, just wrapped in a function to avoid repeating it 3x.
@torch.no_grad()
def split_loss(Xs, Ys):
    emb = C[Xs]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Ys).item()

print("train loss:", split_loss(Xtr, Ytr))
print("dev loss:  ", split_loss(Xdev, Ydev))
# test is intentionally left unevaluated here -- only check it once you're
# truly done tuning (N_HIDDEN, EMB_DIM, LR, ...), or it stops being a fair
# unbiased final check.

# ---- sample from the trained model ----------------------------------------
# Generation is autoregressive: start from an empty context ('.' * block_size),
# run the forward pass to get a probability distribution over the next char,
# SAMPLE one char from that distribution (not argmax -- we want variety, same
# reason makemore Part 1's bigram model sampled instead of always picking the
# most likely next char), append it, slide the context window, repeat until
# the model samples '.' (end token) or we hit a safety cap.
g2 = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    context = [0] * block_size
    out = []
    while True:
        emb = C[torch.tensor([context])]
        emb_flat = emb.view(emb.shape[0], -1)
        h = torch.tanh(emb_flat @ W1 + b1)
        logits = h @ W2 + b2

        # TODO a: turn logits into a probability distribution.
        #   probs = F.softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # TODO b: sample ONE index from that distribution.
        #   ix = torch.multinomial(probs, num_samples=1, generator=g2).item()
        ix = torch.multinomial(probs, num_samples=1, generator=g2).item()

        # TODO c: slide the context window forward, same as build_dataset did.
        #   context = context[1:] + [ix]
        context = context[1:] + [ix]

        out.append(itos[ix])
        if ix == 0 or len(out) > 20:   # '.' = end token, or safety cap
            break
    print("".join(out))