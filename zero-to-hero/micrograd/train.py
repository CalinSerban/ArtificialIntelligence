"""
Lesson 3 (cont.) — Train the MLP with gradient descent
======================================================

We have 4 example inputs and the label we WANT for each (a tiny binary
classification: outputs should be +1 or -1). We'll nudge every weight downhill
until the network's predictions match the targets.

Run:  python3 train.py
"""
import random
from nn import MLP

random.seed(42)

# 4 training examples, each with 3 features.
xs = [
    [2.0,  3.0, -1.0],
    [3.0, -1.0,  0.5],
    [0.5,  1.0,  1.0],
    [1.0,  1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]   # the desired output for each example

model = MLP(3, [4, 4, 1])     # 3 inputs -> 4 -> 4 -> 1 output
print(f"model has {len(model.parameters())} parameters\n")

LEARNING_RATE = 0.3
STEPS = 50

for step in range(STEPS):
    # 1) FORWARD PASS: prediction for every example.
    y_pred = [model(x) for x in xs]

    # 2) LOSS: mean-squared error. Sum of (pred - target)^2 over all examples.
    #    Smaller loss = predictions closer to targets.
    loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, ys))

    # 3) ZERO THE GRADIENTS.  <-- the classic gotcha from the lecture!
    #    Gradients ACCUMULATE (remember the += in every _backward). If you don't
    #    reset them to 0 before each backward pass, this step's gradient gets
    #    added on top of last step's stale gradient -> training misbehaves.
    #
    # TODO: loop over model.parameters() and set each p.grad = 0.0
    for p in model.parameters():
        p.grad = 0.0

    # 4) BACKWARD PASS: fill every parameter's .grad via your engine.
    loss.backward()

    # 5) UPDATE (gradient descent): step each parameter a little DOWNHILL.
    #    p.grad points in the direction that INCREASES loss, so we go negative.
    #
    # TODO: loop over model.parameters() and do  p.data -= LEARNING_RATE * p.grad
    for p in model.parameters():
        p.data -= p.grad * LEARNING_RATE

    if step % 5 == 0 or step == STEPS - 1:
        print(f"step {step:2d}  loss = {loss.data:.6f}")

print("\nfinal predictions (want:  1, -1, -1,  1):")
print([round(model(x).data, 3) for x in xs])
