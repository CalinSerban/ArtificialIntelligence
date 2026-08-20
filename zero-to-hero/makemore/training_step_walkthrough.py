"""
ONE full training step, traced end to end: forward -> loss -> backward -> update.
Same toy setup as forward_pass_walkthrough.py (4 chars, 3 examples).

Run:  ../.venv/bin/python training_step_walkthrough.py
"""
import torch
import torch.nn.functional as F
torch.set_printoptions(precision=3, sci_mode=False)

itos = {0: ".", 1: "a", 2: "b", 3: "c"}
LR = 10.0

xs = torch.tensor([1, 2, 1])   # inputs  = [a, b, a]
ys = torch.tensor([2, 0, 3])   # targets = [b, ., c]
N = len(xs)

g = torch.Generator().manual_seed(1)
W = torch.randn((4, 4), generator=g, requires_grad=True)

print("bigrams we're teaching:", [f"{itos[xs[k].item()]}->{itos[ys[k].item()]}" for k in range(N)])
print("="*60)

# ===== FORWARD =====
xenc   = F.one_hot(xs, 4).float()
logits = xenc @ W                       # pluck each input's row of W
probs  = torch.softmax(logits, dim=1)   # scores -> probabilities
print("FORWARD — probs (one row per example):\n", probs.detach())

# ===== LOSS =====
picked_before = probs[torch.arange(N), ys]      # prob of the TRUE next char
loss = -picked_before.log().mean()
print("\nLOSS")
print("  picked (prob of true char):", picked_before.detach().tolist())
print("  loss = -log(picked).mean() =", loss.item())

# ===== BACKWARD =====
loss.backward()
onehot = F.one_hot(ys, 4).float()
grad_logits_manual = (probs - onehot) / N       # the clean formula
print("\nBACKWARD")
print("  grad on logits, per example  = (probs - onehot)/N:")
for k in range(N):
    print(f"    ex{k} ({itos[xs[k].item()]}->{itos[ys[k].item()]}):",
          [round(v, 3) for v in grad_logits_manual[k].tolist()])
print("  grad on W (autograd):")
print(W.grad)
print("  note row 'a' (index 1) = sum of ex0 + ex2 (both had input 'a') -> grads ACCUMULATE")

# ===== UPDATE =====
with torch.no_grad():
    W_new = W - LR * W.grad
print("="*60)
print(f"UPDATE — W_new = W - {LR} * W.grad")

# ===== DID IT LEARN? re-run forward with W_new =====
probs_after  = torch.softmax(xenc @ W_new, dim=1)
picked_after = probs_after[torch.arange(N), ys]
loss_after   = -picked_after.log().mean()
print("\nDID THE TRUE-CHAR PROBABILITIES GO UP?")
for k in range(N):
    b, a = picked_before[k].item(), picked_after[k].item()
    print(f"  ex{k} ({itos[xs[k].item()]}->{itos[ys[k].item()]}): P(true) {b:.3f} -> {a:.3f}   {'UP' if a>b else 'down'}")
print(f"\n  loss {loss.item():.3f} -> {loss_after.item():.3f}   (should DROP)")
