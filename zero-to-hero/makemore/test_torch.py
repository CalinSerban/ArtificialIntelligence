import torch
import matplotlib
matplotlib.use("Agg")   # headless backend: render to a file instead of a live window
import matplotlib.pyplot as plt

generator = torch.Generator().manual_seed(324234325)

# use a LOT of samples -- the bell shape only emerges clearly with many draws.
# a 40x40 grid is only 1600 values; bump it up so bins fill in smoothly.
t = torch.randn(10000000, generator=generator)
print("t shape:", tuple(t.shape))

# TODO: plot a histogram of t's values to see the Gaussian bell shape.
#   plt.hist(t, bins=100)
#   (bins=100 means the range gets split into 100 buckets; each bar's
#    height = how many of the 100,000 values landed in that bucket)
plt.hist(t, bins=100)

plt.title("torch.randn(100_000) -- value distribution")
plt.xlabel("value")
plt.ylabel("count")
plt.savefig("randn_plot.png", dpi=150, bbox_inches="tight")
print("saved to randn_plot.png")
