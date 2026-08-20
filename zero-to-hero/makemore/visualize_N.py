"""
Visualize the bigram count matrix N as a heatmap.
Each cell (i, j): the bigram "ij" and how many times char j followed char i.

Run:  ../.venv/bin/python visualize_N.py   ->  writes bigram_matrix.png
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# --- rebuild N (same as bigram_counts.py, kept self-contained) --------------
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

# --- design tokens ----------------------------------------------------------
INK      = "#1a1a1a"   # primary text
MUTED    = "#8a8a8a"   # recessive text
SURFACE  = "#ffffff"
CMAP     = "Blues"     # single-hue sequential: pale=rare, deep=common

# sqrt normalization (gamma=0.5) so mid-frequency pairs don't wash out to pale
norm = PowerNorm(gamma=0.5, vmin=0, vmax=int(N.max()))

# --- draw -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 18), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
im = ax.imshow(N, cmap=CMAP, norm=norm)

for i in range(VOCAB):
    for j in range(VOCAB):
        bigram = itos[i] + itos[j]
        count  = N[i, j].item()
        # flip text color for contrast: white on deep cells, dark on pale ones
        t = norm(count)                      # normalized 0..1 for this cell
        txt_color = "white" if t > 0.5 else INK
        ax.text(j, i - 0.18, bigram, ha="center", va="center",
                color=txt_color, fontsize=8)
        ax.text(j, i + 0.22, str(count), ha="center", va="center",
                color=txt_color, fontsize=7)

# axis labels = the characters, ticks recessive
ax.set_xticks(range(VOCAB)); ax.set_yticks(range(VOCAB))
ax.set_xticklabels([itos[i] for i in range(VOCAB)], color=MUTED)
ax.set_yticklabels([itos[i] for i in range(VOCAB)], color=MUTED)
ax.xaxis.tick_top()
ax.set_xlabel("next character (j)", color=INK, fontsize=12)
ax.xaxis.set_label_position("top")
ax.set_ylabel("current character (i)", color=INK, fontsize=12)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

ax.set_title("Bigram counts — how often character j follows character i\n"
             "(row 0 / col 0 = '.' start/end token)",
             color=INK, fontsize=15, pad=28)

# slim colorbar as the magnitude legend
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("occurrences (sqrt-scaled color)", color=MUTED)
cbar.ax.tick_params(colors=MUTED)
cbar.outline.set_visible(False)

fig.tight_layout()
fig.savefig("bigram_matrix.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote bigram_matrix.png")

# a couple of quick reads from the data
row_dot = N[0].clone(); row_dot[0] = 0
print("most common STARTING letter:", itos[row_dot.argmax().item()],
      f"({row_dot.max().item()} names)")
col_dot = N[:, 0].clone(); col_dot[0] = 0
print("most common ENDING letter:  ", itos[col_dot.argmax().item()],
      f"({col_dot.max().item()} names)")
