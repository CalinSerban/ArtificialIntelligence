"""
makemore Part 1c — the SAME bigram model, as a neural network
==============================================================
One 27x27 weight matrix W, trained by gradient descent to minimize NLL.
It should converge to NLL ~2.45 and its probabilities should match the
counting model's P.

Reuses everything you know:
  - the forward pass turns W into probabilities (softmax)
  - the loss is your NLL, vectorized
  - the training loop is your micrograd 5 steps, now in PyTorch

Fill the TODOs. Run:  ../.venv/bin/python neural_net.py
"""
import torch
import torch.nn.functional as F

# --- vocab ------------------------------------------------------------------
words = open("names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

# --- 1) build the training set: every bigram as (input x -> target y) [GIVEN]
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])   # input  = current char index
        ys.append(stoi[ch2])   # target = next char index
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f"{num} bigram examples\n")

# --- 2) the network: a single 27x27 weight matrix [GIVEN] -------------------
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)   # random start

# --- 3) TRAINING LOOP -------------------------------------------------------
for k in range(200):
    # ===== FORWARD PASS =====
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    probs = F.softmax(logits, dim=1)
    loss = -probs[torch.arange(num), ys].log().mean()

    # ===== BACKWARD PASS =====
    # TODO f: reset gradients, then backprop.
    W.grad = None
    loss.backward()

    # ===== UPDATE (gradient descent) =====
    # TODO g: step W downhill. Use a BIG learning rate here (~50).
    #   W.data += -50 * W.grad
    W.data += -50 * W.grad

    if k % 20 == 0:
        print(f"step {k:3d}  loss = {loss.item():.4f}")

# --- 4) COMPARE to the counting model [GIVEN] -------------------------------
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        N[stoi[a], stoi[b]] += 1
P_count  = N.float() / N.float().sum(1, keepdim=True)
P_neural = torch.softmax(W, dim=1)              # softmax of learned weights
print("\nmax |P_count - P_neural| =", (P_count - P_neural).abs().max().item(),
      "  <- small means they learned the same thing")

# sample 5 names from the NEURAL model (same sampler as the count model)
g2 = torch.Generator().manual_seed(2147483647)
print("\nnames from the neural net:")
for _ in range(5):
    out, ix = [], 0
    while True:
        p = P_neural[ix]
        ix = torch.multinomial(p, 1, replacement=True, generator=g2).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("  ", "".join(out))
