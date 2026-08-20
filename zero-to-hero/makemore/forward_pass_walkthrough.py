"""
Forward pass of the neural bigram model, traced step by step.
Uses a TINY toy setup (4 chars, 3 examples) so every tensor is readable.

The 4 forward-pass lines (same as neural_net.py, just small):
    xenc   = F.one_hot(xs, vocab).float()      # 1) index -> one-hot
    logits = xenc @ W                          # 2) pluck a row of W (scores)
    counts = logits.exp()                      # 3) make positive  ]
    probs  = counts / counts.sum(1, keepdim=True)  # 4) normalize   ]-> softmax

Run:  ../.venv/bin/python forward_pass_walkthrough.py

>>> PLAYGROUND:
    - change xs to other char indices and watch which row of W gets plucked.
    - replace steps 3+4 with  probs = torch.softmax(logits, dim=1)  -> same result.
    - grow the toy: vocab=27 and real xs/ys to match neural_net.py.
"""
import torch
import torch.nn.functional as F
torch.set_printoptions(precision=3, sci_mode=False)

# toy vocab: .=0  a=1  b=2  c=3   (real model uses 27)
VOCAB = 4
g = torch.Generator().manual_seed(1)
W = torch.randn((VOCAB, VOCAB), generator=g)   # learned lookup table of log-counts
print("W (4x4 weights):\n", W, "\n")

xs = torch.tensor([1, 2, 1])   # inputs  = current chars [a, b, a]
ys = torch.tensor([2, 0, 3])   # targets = true next chars [b, ., c]
print("xs (inputs):", xs.tolist(), "  ys (targets):", ys.tolist(), "\n")

print("="*50, "\nSTEP 1 — one-hot encode  (index -> vector with a single 1)")
xenc = F.one_hot(xs, num_classes=VOCAB).float()
print("xenc", tuple(xenc.shape), ":\n", xenc, "\n")

print("="*50, "\nSTEP 2 — logits = xenc @ W  (one-hot @ W just PLUCKS a row of W)")
logits = xenc @ W
print("logits", tuple(logits.shape), ":\n", logits)
print("logits[0] == W[1]?  input was char 1 ->", torch.allclose(logits[0], W[1]), "\n")

print("="*50, "\nSTEP 3 — counts = logits.exp()  (map any real number -> positive)")
counts = logits.exp()
print("counts", tuple(counts.shape), ":\n", counts, "\n")

print("="*50, "\nSTEP 4 — probs = counts / counts.sum(1, keepdim=True)  (normalize rows)")
probs = counts / counts.sum(1, keepdim=True)
print("probs", tuple(probs.shape), ":\n", probs)
print("row sums:", probs.sum(1).tolist(), "  (STEP 3+4 together = softmax)\n")

print("="*50, "\nLOSS — prob assigned to the TRUE next char, -log, averaged")
picked = probs[torch.arange(len(xs)), ys]
print("probs[arange, ys] =", picked.tolist())
print("loss = -picked.log().mean() =", (-picked.log().mean()).item())
