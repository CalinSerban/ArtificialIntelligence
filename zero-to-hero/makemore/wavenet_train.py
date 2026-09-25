"""
makemore Part 5 — Step 3: train the real WaveNet
=====================================================
Same overall shape as mlp_train.py (train/dev/test split, minibatch SGD,
periodic loss printing, final split_loss eval, autoregressive sampling) --
but now the "model" is the hierarchical Sequential stack from
wavenet_layers.py instead of 5 loose W/b tensors, so the forward pass
collapses to one line: `model(Xb)`.

One genuinely new wrinkle: BatchNorm1d behaves differently at train vs.
eval time (current-minibatch stats vs. running average) -- so before
evaluating on dev/test you must flip every BatchNorm1d into eval mode
(model.eval()), and flip back before resuming training (model.train()).
That's exactly what the .train()/.eval() TODOs (j/k) in wavenet_layers.py
are for -- fill those first if you haven't.

Fill the TODOs, then run:  ../.venv/bin/python wavenet_train.py
"""
import random
import torch
import torch.nn.functional as F
from wavenet_dataset import build_dataset, stoi, itos, block_size
from wavenet_layers import Sequential, Embedding, FlattenConsecutive, Linear, BatchNorm1d, Tanh

words = open("names.txt").read().splitlines()

# same 80/10/10 split-by-word logic as mlp_train.py
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print("train:", Xtr.shape[0], " dev:", Xdev.shape[0], " test:", Xte.shape[0])

VOCAB = 27
N_EMBD = 24
N_HIDDEN = 128
BATCH_SIZE = 32
STEPS = 20000
LR = 0.1

g = torch.Generator().manual_seed(2147483647)

# same hierarchical stack as the wavenet_layers.py self-test: block_size=8
# characters -> fuse pairs three times (8 -> 4 -> 2 -> 1) -> classify.
model = Sequential([
    Embedding(VOCAB, N_EMBD, generator=g),
    FlattenConsecutive(2), Linear(N_EMBD * 2, N_HIDDEN, bias=False, generator=g), BatchNorm1d(N_HIDDEN), Tanh(),
    FlattenConsecutive(2), Linear(N_HIDDEN * 2, N_HIDDEN, bias=False, generator=g), BatchNorm1d(N_HIDDEN), Tanh(),
    FlattenConsecutive(2), Linear(N_HIDDEN * 2, N_HIDDEN, bias=False, generator=g), BatchNorm1d(N_HIDDEN), Tanh(),
    Linear(N_HIDDEN, VOCAB, generator=g),
])

# TODO a: every parameter needs requires_grad=True, same as mlp_train.py --
# except now the flat list comes from model.parameters() instead of a
# hand-written `params = [C, W1, b1, W2, b2]`.
#   for p in model.parameters(): p.requires_grad = True
for p in model.parameters():
    p.requires_grad = True

print("param count:", sum(p.nelement() for p in model.parameters()))

lossi = []

for step in range(STEPS):
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # TODO b: forward pass -- one line now, `model(Xb)` gives you logits
    # directly (Embedding->FlattenConsecutive->Linear->BatchNorm1d->Tanh
    # chain all happens inside Sequential.__call__).
    #   logits = model(Xb)
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    lr = LR if step < STEPS * 0.75 else 0.01
    for p in model.parameters():
        p.data -= lr * p.grad

    lossi.append(loss.item())
    if step % 2000 == 0 or step == STEPS - 1:
        print(f"  step {step:6d}  minibatch loss {loss.item():.4f}")

# TODO c: before evaluating on dev/test, switch every BatchNorm1d to eval
# mode -- otherwise it'd normalize using THIS split_loss call's batch
# stats (fine for train, wrong for a fair dev/test read) instead of the
# running averages accumulated during training.
    model.eval()

@torch.no_grad()
def split_loss(Xs, Ys):
    logits = model(Xs)
    return F.cross_entropy(logits, Ys).item()

model.eval()

print("train loss:", split_loss(Xtr, Ytr))
print("dev loss:  ", split_loss(Xdev, Ydev))
print("(Part 2 flat-MLP baseline was  train 2.1676 / dev 2.1932 -- lower is better)")

# ---- sample from the trained model -----------------------------------
# same autoregressive loop as mlp_train.py, just with block_size=8 context
# and model(...) instead of the manual C/W1/b1/W2/b2 forward pass.
g2 = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    context = [0] * block_size
    out = []
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g2).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0 or len(out) > 20:
            break
    print("".join(out))
