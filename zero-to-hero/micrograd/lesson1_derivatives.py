"""
Lesson 1 — What a derivative actually IS (in code)
===================================================

You learned from 3Blue1Brown that a derivative df/dx measures:
    "if I nudge x by a tiny amount, how much does f move, and in what direction?"

Backprop is nothing more than applying that idea, over and over, through a
network. Before we build the engine, let's SEE a derivative as a number.

Run me with:   python3 lesson1_derivatives.py
"""

# ---------------------------------------------------------------------------
# Part A — the numerical derivative (the definition, in code)
# ---------------------------------------------------------------------------
# The math definition:   f'(x) = lim(h->0) [ f(x+h) - f(x) ] / h
# We can't do "limit to zero" on a computer, but we can pick a tiny h and peek.

def f(x):
    return 3 * x**2 - 4 * x + 5

h = 0.000001
x = 3.0

# Rise over run: how much f changed, divided by how much we nudged x.
numerical_slope = (f(x + h) - f(x)) / h
print("f(3)            =", f(x))
print("numerical slope at x=3 :", numerical_slope)

# The analytic derivative of f is  f'(x) = 6x - 4.
# At x=3 that's 6*3 - 4 = 14.  Check that your number above is ~14.
print("analytic slope at x=3  :", 6 * x - 4)
print()

# ---------------------------------------------------------------------------
# Part B — a derivative with MULTIPLE inputs (this is the real setup)
# ---------------------------------------------------------------------------
# A neural net is a function of many inputs. The "gradient" is just the list of
# derivatives, one per input: how sensitive the output is to each input.
#
# Let:   d = a*b + c
# Then the three partial derivatives are:
#   dd/da = b      (nudging a moves d by b's worth)
#   dd/db = a
#   dd/dc = 1
#
# >>> YOUR TASK <<<
# Fill in the three numerical derivatives below using the same
# "nudge and measure" trick from Part A. Nudge ONE variable at a time.

a = 2.0
b = -3.0
c = 10.0

def d_of(a, b, c):
    return a * b + c

# TODO: replace each None with a numerical-derivative expression.
# Hint for dd_da:  ( d_of(a + h, b, c) - d_of(a, b, c) ) / h
dd_da = (d_of(a + h, b, c) - d_of(a, b, c)) / h
dd_db = (d_of(a, b + h, c) - d_of(a, b, c)) / h
dd_dc = (d_of(a, b, c + h) - d_of(a, b, c)) / h

print("dd/da  (expect  b = -3.0):", dd_da)
print("dd/db  (expect  a =  2.0):", dd_db)
print("dd/dc  (expect      1.0):", dd_dc)
