"""
Lesson 2 — Build the micrograd engine yourself
===============================================

You watched Karpathy build the `Value` object. Now rebuild it from this
scaffold. The class structure, the `__add__` op, and `backward()` (the
topological sort) are given as WORKED REFERENCES. Your job is to fill in the
local gradients for the other operations — the `_backward` closures marked
`# TODO`.

The rule for every op, no exceptions:
  1. Compute the forward output value.
  2. Wrap it in a new Value, recording its parents (_prev) and op name (_op).
  3. Define a _backward() closure that pushes the output's gradient (out.grad)
     back to each parent, multiplied by the LOCAL derivative of this op.
     Always use  +=  (a value used twice accumulates gradient — chain rule).

When done:   python3 test_engine.py
"""


class Value:
    """A single scalar value that remembers how it was produced."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0                 # d(final_output)/d(self), starts at 0
        self._backward = lambda: None   # how to push grad to parents; set per-op
        self._prev = set(_children)     # the Values that produced this one
        self._op = _op                  # label, for debugging/graph drawing

    # ---- WORKED REFERENCE: addition ------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> just pass grad through
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # ---- YOUR TURN: multiplication -------------------------------------
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: d(out)/d(self) = other   d(out)/d(other) = self
            # (remember out = self * other)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # ---- YOUR TURN: power (x ** k, k is a plain number) ----------------
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # TODO: d(out)/d(self) for out = self**other  is  other*self**(other-1)
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    # ---- YOUR TURN: tanh (a squashing nonlinearity) --------------------
    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO: d(tanh(x))/dx = 1 - tanh(x)**2 = 1 - t**2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    # ---- WORKED REFERENCE: backpropagation -----------------------------
    def backward(self):
        # Build a topological order: every node appears AFTER all its parents.
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # Seed: derivative of the output w.r.t. itself is 1.
        self.grad = 1.0
        # Walk in reverse: apply each node's local backward, propagating grads.
        for node in reversed(topo):
            node._backward()

    # ---- convenience ops (given, derived from the core ops) ------------
    def __neg__(self):            return self * -1
    def __radd__(self, other):    return self + other
    def __sub__(self, other):     return self + (-other)
    def __rsub__(self, other):    return other + (-self)
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other):return other * self ** -1
    def __repr__(self):           return f"Value(data={self.data}, grad={self.grad})"
