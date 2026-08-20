"""
Compare hidden-layer health: gain=1 vs gain=5/3 (the tanh-calibrated constant).
std(W) = gain / sqrt(fan_in)

Fill the TODOs, then run:  ../.venv/bin/python compare_init.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlp_dataset import build_dataset, block_size

words = open("names.txt").read().splitlines()
X, Y = build_dataset(words)

VOCAB, EMB_DIM, N_HIDDEN = 27, 2, 100
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((VOCAB, EMB_DIM), generator=g)
emb_flat = C[X].view(X.shape[0], -1)
fan_in = block_size * EMB_DIM

def make_h(gain):
    W1 = torch.randn((fan_in, N_HIDDEN), generator=g) * gain / (fan_in ** 0.5)
    b1 = torch.randn(N_HIDDEN, generator=g) * 0.01
    return torch.tanh(emb_flat @ W1 + b1)

h_gain1  = make_h(gain=1.0)
h_gain53 = make_h(gain=5/3)

for name, h in [("gain=1", h_gain1), ("gain=5/3", h_gain53)]:
    print(f"--- {name} ---")

    # TODO a: saturation fraction -- of ALL (example, hidden-unit) entries,
    # what fraction have |h| > 0.99?
    #   sat_mask = h.abs() > 0.99
    #   sat_fraction = sat_mask.float().mean()
    sat_mask = h.abs() > 0.99
    sat_fraction = sat_mask.float().mean()
    print("  saturation fraction:", sat_fraction)

    # TODO b: dead neurons -- how many of the N_HIDDEN COLUMNS are
    # saturated for EVERY example (torch.all along the example axis, dim=0)?
    #   dead_per_column = sat_mask.all(dim=0)   # shape (N_HIDDEN,), one bool per neuron
    #   num_dead = dead_per_column.sum()
    dead_per_column = sat_mask.all(dim=0)
    num_dead = dead_per_column.sum()
    print("  dead neurons (out of", N_HIDDEN, "):", num_dead)
    print()

# ===========================================================================
# ONE hidden layer barely showed a difference. Now stack 20 tanh layers back
# to back, each layer's output feeding the next, and watch how the signal's
# spread (std) evolves with depth. This is the setting gain=5/3 is actually
# calibrated for.
# ===========================================================================
N_LAYERS = 40
WIDTH = 100          # every layer same width, so fan_in stays constant = WIDTH
BATCH = 1000

def run_stack(gain, seed=42):
    gg = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, WIDTH, generator=gg)   # synthetic input, unit variance
    stds = [x.std().item()]
    sat_fracs = [(x.abs() > 0.99).float().mean().item()]

    for layer in range(N_LAYERS):
        W = torch.randn((WIDTH, WIDTH), generator=gg) * gain / (WIDTH ** 0.5)
        b = torch.randn(WIDTH, generator=gg) * 0.01

        # TODO c: forward pass through this one layer -- linear, then tanh.
        #   x = torch.tanh(x @ W + b)
        x = torch.tanh(x @ W + b)

        stds.append(x.std().item())
        sat_fracs.append((x.abs() > 0.99).float().mean().item())
    return stds, sat_fracs

stds_g1,  sat_g1  = run_stack(gain=1.0)
stds_g53, sat_g53 = run_stack(gain=5/2)

print(f"{'layer':>5}  {'std g=1':>10}  {'std g=5/3':>10}  {'sat% g=1':>10}  {'sat% g=5/3':>10}")
for i in range(N_LAYERS + 1):
    print(f"{i:5d}  {stds_g1[i]:10.4f}  {stds_g53[i]:10.4f}  "
          f"{sat_g1[i]*100:9.1f}%  {sat_g53[i]*100:9.1f}%")

# TODO d: plot std vs layer depth for both gains, to see the curves diverge.
plt.plot(stds_g1, label="gain=1")
plt.plot(stds_g53, label="gain=5/3")

plt.xlabel("layer depth")
plt.ylabel("std of activations")
plt.title(f"Activation std across {N_LAYERS} stacked tanh layers")
plt.legend()
plt.savefig("deep_init_demo.png", dpi=150, bbox_inches="tight")
print("saved to deep_init_demo.png")
