"""
XOR -- the classic "why nonlinearity" example.
=================================================
    (0,0) -> 0
    (0,1) -> 1
    (1,0) -> 1
    (1,1) -> 0
The two "1" points and two "0" points sit at opposite corners of a square --
no single straight line separates them. NOT linearly separable. A stack of
pure linear layers (no matter how deep) collapses into one linear layer, so
it can never solve this. One hidden layer WITH tanh can.

Same architecture, same data, same steps -- once with tanh, once without.
Watch the loss curves diverge.

Fill the TODOs, then run:  ../.venv/bin/python xor_nn.py
"""
import torch

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([0., 1., 1., 0.])

H = 4          # hidden units -- small on purpose, XOR is tiny
LR = 0.01
STEPS = 3000

def train(use_tanh, steps=STEPS, seed=0):
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn((2, H), generator=g, requires_grad=True)
    b1 = torch.randn(H, generator=g, requires_grad=True)
    W2 = torch.randn((H, 1), generator=g, requires_grad=True)
    b2 = torch.randn(1, generator=g, requires_grad=True)
    params = [W1, b1, W2, b2]

    for step in range(steps):
        # TODO a: forward pass.
        #   pre_h = X @ W1 + b1
        #   h = torch.tanh(pre_h) if use_tanh else pre_h
        #   pred = (h @ W2 + b2).squeeze()
        pre_h = X @ W1 + b1
        h = torch.tanh(pre_h) if use_tanh else pre_h
        pred = (h @ W2 + b2).squeeze()

        # TODO b: loss -- mean squared error between pred and Y.
        #   loss = ((pred - Y) ** 2).mean()
        loss = ((pred - Y) ** 2).mean()

        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:4d}  pred {pred.detach().tolist()}  loss {loss.item():.4f}")

        # TODO c: backward + update (same pattern as neural_net.py / training_step_walkthrough.py)
        #   for p in params: p.grad = None
        #   loss.backward()
        #   with torch.no_grad():
        #       for p in params: p -= LR * p.grad
        for p in params:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= LR * p.grad
        
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    return pred, loss

print("=== WITH tanh (nonlinear) ===")
pred_tanh, loss_tanh = train(use_tanh=True)
print("final predictions:", [round(v, 3) for v in pred_tanh.detach().tolist()],
      " (target:", Y.tolist(), ")")

print()
print("=== WITHOUT tanh (pure linear) ===")
pred_lin, loss_lin = train(use_tanh=False)
print("final predictions:", [round(v, 3) for v in pred_lin.detach().tolist()],
      " (target:", Y.tolist(), ")")
