"""
makemore Part 1a — the COUNTING bigram model
============================================
A bigram model predicts the next character from ONLY the current one.
We build it by pure counting: how often does 'b' follow 'a', etc.

Special token '.' marks both the START and END of a word, so the model can
learn which letters tend to start names and which tend to end them.

Fill in the TODOs, then run:  ../.venv/bin/python bigram_counts.py
(or activate the venv first:  source ../.venv/bin/activate)
"""
import torch

# ---------------------------------------------------------------------------
# 0) Load the data
# ---------------------------------------------------------------------------
words = open("names.txt").read().splitlines()
print(f"{len(words)} names. first five: {words[:5]}")
print(f"shortest: {min(len(w) for w in words)}  longest: {max(len(w) for w in words)}\n")

# ---------------------------------------------------------------------------
# 1) Build the vocabulary + char<->index maps  (GIVEN — this part is fiddly)
# ---------------------------------------------------------------------------
chars = sorted(set("".join(words)))     # the 26 letters a..z
stoi = {ch: i + 1 for i, ch in enumerate(chars)}   # a->1, b->2, ... z->26
stoi["."] = 0                            # the special start/end token -> 0
itos = {i: ch for ch, i in stoi.items()} # reverse map: index -> char
VOCAB = len(stoi)                        # 27
print(f"vocab size: {VOCAB}  (26 letters + '.')\n")

# ---------------------------------------------------------------------------
# 2) Build the count matrix N  (VOCAB x VOCAB)
#    N[i, j] = how many times character j follows character i.
# ---------------------------------------------------------------------------
N = torch.zeros((VOCAB, VOCAB), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]        # e.g. emma -> . e m m a .
    for ch1, ch2 in zip(chs, chs[1:]):   # consecutive pairs: (.,e)(e,m)(m,m)(m,a)(a,.)
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        N[i1, i2] += 1

print("count for ('.', a..e):", N[0, 1:6].tolist(), "(names starting with a,b,c,d,e)")
print("count for ('a','.'):  ", N[1, 0].item(), "(names where 'a' is the last letter)\n")

# ---------------------------------------------------------------------------
# 3) Normalize rows -> probability matrix P
#    Each row i must sum to 1: "given char i, prob of each next char".
#    Remember the keepdim lesson!
# ---------------------------------------------------------------------------
# TODO: P = N (as float) divided by its per-row sum, keeping dims so rows sum to 1.
# Hint: P = N.float(); P = P / P.sum(1, keepdim=True)
P = N.float()
P = P / P.sum(1, keepdim = True)

# sanity check — every row should sum to 1.0
# (uncomment once P is set)
assert torch.allclose(P.sum(1), torch.ones(VOCAB)), "rows don't sum to 1 — keepdim bug?"
print("row-sum check passed ✓\n")

# ---------------------------------------------------------------------------
# 4) Sample new names from P  (GIVEN — introduces torch.multinomial)
#    Start at '.', repeatedly sample the next char from the current row of P,
#    stop when we sample '.' again.
# ---------------------------------------------------------------------------
def sample_name(g):
    out = []
    ix = 0                                # start token '.'
    while True:
        probs = P[ix]                     # row = distribution over next char
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:                       # sampled '.', word is done
            break
        out.append(itos[ix])
    return "".join(out)

# generate a few (only runs once P is real)
if P is not None:
    g = torch.Generator().manual_seed(2147483647)
    print("sampled names:")
    for _ in range(5):
        print("  ", sample_name(g))
