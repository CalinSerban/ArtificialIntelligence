"""
Checks your engine.py by comparing its analytic gradients (from backward())
against numerical gradients (nudge-and-measure, like Lesson 1).
No PyTorch needed. Run:  python3 test_engine.py
"""
from engine import Value


def numerical_grad(f, inputs, i, h=1e-6):
    """Gradient of scalar f w.r.t. inputs[i], by nudging."""
    up = list(inputs); up[i] += h
    dn = list(inputs); dn[i] -= h
    print(f"up[i]={up[i]:.6f} dn[i]={dn[i]:.6f} f(up)={f(*up):.6f} f(dn)={f(*dn):.6f}")
    return (f(*up) - f(*dn)) / (2 * h)


def check(name, expr, plain, inputs):
    """expr: fn of Values (uses your engine). plain: same math on floats."""
    vals = [Value(x) for x in inputs]
    out = expr(*vals)
    out.backward()
    ok = True
    for i, v in enumerate(vals):
        ng = numerical_grad(plain, inputs, i)
        diff = abs(v.grad - ng)
        status = "ok " if diff < 1e-4 else "FAIL"
        if diff >= 1e-4:
            ok = False
        print(f"  [{status}] {name}: d/dx{i}  analytic={v.grad:+.5f}  numerical={ng:+.5f}")
    return ok


import math

tests = [
    # (name, uses-Value version, plain-float version, input values)
    ("mul",      lambda a, b: a * b,                 lambda a, b: a * b,                 [2.0, -3.0]),
    ("pow",      lambda a: a ** 3,                    lambda a: a ** 3,                    [2.0]),
    ("div",      lambda a, b: a / b,                  lambda a, b: a / b,                  [4.0, 2.0]),
    ("tanh",     lambda a: a.tanh(),                  lambda a: math.tanh(a),             [0.7]),
    # a small "neuron": tanh(w1*x1 + w2*x2 + b)
    ("neuron",   lambda w1, x1, w2, x2, b: (w1*x1 + w2*x2 + b).tanh(),
                 lambda w1, x1, w2, x2, b: math.tanh(w1*x1 + w2*x2 + b),
                 [1.0, 2.0, -3.0, 0.5, 0.1]),
]

all_ok = True
for name, expr, plain, inputs in tests:
    all_ok &= check(name, expr, plain, inputs)

print("\n" + ("ALL TESTS PASSED — your engine backprops correctly! 🎉"
              if all_ok else "Some tests failed — check the _backward with FAIL above."))
