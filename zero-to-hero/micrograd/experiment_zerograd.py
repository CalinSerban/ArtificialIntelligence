"""
Experiment — why does zero_grad matter?
=======================================

Same data, same seed, same learning rate, same number of steps.
The ONLY difference between the two runs is whether we reset gradients to 0
before each backward pass. Watch what accumulating stale gradients does.

Run:  python3 experiment_zerograd.py
"""
import random
from nn import MLP

xs = [
    [2.0,  3.0, -1.0],
    [3.0, -1.0,  0.5],
    [0.5,  1.0,  1.0],
    [1.0,  1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

LEARNING_RATE = 0.05
STEPS = 50


def train(zero_grad: bool):
    random.seed(42)          # same seed -> both runs start from the SAME weights
    model = MLP(3, [4, 4, 1])
    history = []
    for step in range(STEPS):
        # forward + loss
        y_pred = [model(x) for x in xs]
        loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, ys))

        # the one line under test:
        if zero_grad:
            for p in model.parameters():
                p.grad = 0.0

        # backward + update
        loss.backward()
        for p in model.parameters():
            p.data -= LEARNING_RATE * p.grad

        history.append(loss.data)
    return history, [round(model(x).data, 2) for x in xs]


good_hist, good_pred = train(zero_grad=True)
bad_hist,  bad_pred  = train(zero_grad=False)

print(f"{'step':>4} | {'WITH zero_grad':>16} | {'WITHOUT zero_grad':>18}")
print("-" * 46)
for step in range(0, STEPS, 5):
    print(f"{step:>4} | {good_hist[step]:>16.6f} | {bad_hist[step]:>18.6f}")

print("\ntargets:              ", ys)
print("WITH zero_grad  preds:", good_pred)
print("WITHOUT         preds:", bad_pred)
