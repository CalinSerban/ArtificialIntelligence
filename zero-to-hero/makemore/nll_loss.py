"""
makemore Part 1b — score the model with Negative Log-Likelihood (NLL)
=====================================================================
NLL = the average of -log(P[bigram]) over every bigram in the dataset.
Lower = better. A perfect model would score 0.

You write the loop. Run:  ../.venv/bin/python nll_loss.py
"""
import torch

# --- rebuild P (counts -> probabilities), same as before --------------------
words = open("names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}
VOCAB = len(stoi)

N = torch.zeros((VOCAB, VOCAB), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        N[stoi[a], stoi[b]] += 1
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

def score(word):
    """
    Compute the average negative log-likelihood of a word under the bigram model.
    """
    chs = ['.'] + list(word) + ['.']
    log_likelihood = 0.0
    n = 0
    for ch1, ch2 in zip(chs, chs[1:]):
        i1, i2 = stoi[ch1], stoi[ch2]
        prob = P[i1, i2]
        log_likelihood += torch.log(prob)
        n += 1
    nl = -log_likelihood
    avg_nll = nl / n
    print(f"average negative log-likelihood: {avg_nll.item():.4f} for {word}")

log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        i1, i2 = stoi[ch1], stoi[ch2]
        prob = P[i1,i2]
        log_likelihood += torch.log(prob)
        n += 1
nl = -log_likelihood
avg_nll = nl / n
print(f"average negative log-likelihood: {avg_nll.item():.4f} over {n} bigrams")

score("xcrq")
