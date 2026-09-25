"""
makemore Part 5 — Step 1: a mini torch.nn, built by hand
=============================================================
Before doing the WaveNet hierarchical-fusion trick, refactor the flat MLP
(mlp_train.py) into composable LAYER OBJECTS -- same math you've already
written by hand several times, just wrapped so layers can be chained
generically (a Sequential of arbitrary layers) instead of one hardcoded
forward pass. This is exactly the shape of real torch.nn.Module/Linear/etc.

Each layer below is a plain Python class with:
  - __call__(self, x)   -- the forward pass. Stores its output as self.out
                           (handy for later inspecting activations layer by
                           layer, e.g. to check for saturation/dead units).
  - .parameters()       -- returns the list of tensors that need gradients,
                           so a training loop can do `for p in model.parameters()`
                           instead of hand-listing every W/b like mlp_train.py did.

BatchNorm1d is the one genuinely NEW piece (Part 3 territory, which we
skipped) -- explained inline below.

Fill the TODOs, then run:  ../.venv/bin/python wavenet_layers.py
(runs a small self-test at the bottom, not real training yet)
"""
import torch

# -----------------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        # /fan_in**0.5 scaling (not a bare randn like mlp_loss.py used for
        # W1) keeps activations from exploding as layers stack deeper --
        # more on this "Kaiming init" idea once we hit Part 3 material.
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        # TODO a: the forward pass -- exactly the matmul+bias you've written
        # by hand in mlp_loss.py/mlp_train.py (h = emb_flat @ W1 + b1), just
        # using self.weight/self.bias instead of loose W/b variables.
        #   out = x @ self.weight
        #   if self.bias is not None:
        #       out = out + self.bias
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


# -----------------------------------------------------------------------
class BatchNorm1d:
    """
    Normalizes each activation (each column of x) to mean 0, std 1 across
    the BATCH, then applies a learnable affine transform (gamma * xhat +
    beta) so the network can un-normalize if that's actually better for a
    given unit. Why bother: earlier you saw tanh saturate/die when
    pre-activations drift too large/small (dying ReLU, XOR divergence).
    BatchNorm keeps pre-activations in a well-behaved range throughout
    training, layer by layer, instead of hoping init alone is enough.

    Train vs eval matters here: at TRAINING time we normalize using the
    CURRENT MINIBATCH's mean/var (batch-dependent, noisy -- also acts as a
    mild regularizer). At EVAL/inference time we can't do that (might be
    predicting on a single example, no "batch" to compute stats from, and
    we want deterministic output) -- so during training we also maintain a
    running average of mean/var, and use THAT at eval time instead.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # learnable affine params
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # running stats (not learned by gradient descent -- updated manually
        # via a running average each forward pass during training)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # x is (batch, channels) coming straight out of a flat Linear, but
        # (batch, time, channels) coming out of a Linear that's sandwiched
        # between two FlattenConsecutive stages (WaveNet's hierarchical
        # layers below). Either way, every dim EXCEPT the last (channels)
        # should get averaged over -- that's "everything that isn't a
        # feature" for a 2D input that's just dim 0, for a 3D input it's
        # dims (0, 1).
        #   TODO h: dim = (0,) if x.ndim == 2 else (0, 1)
        dim = (0,) if x.ndim == 2 else (0, 1)

        if self.training:
            # TODO b: compute THIS BATCH's mean and variance over `dim`
            # (keepdim=True so the result still broadcasts against x).
            #   xmean = x.mean(dim, keepdim=True)
            #   xvar = x.var(dim, keepdim=True)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        # TODO c: normalize x to mean 0 / std 1 using xmean/xvar, then apply
        # the learnable affine transform.
        #   xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        #   out = self.gamma * xhat + self.beta
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                # TODO d: update the running stats with an exponential
                # moving average: new_running = (1-momentum)*old_running +
                # momentum*this_batch_stat. This is what eval-mode will use.
                #   self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                #   self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# -----------------------------------------------------------------------
class Tanh:
    def __call__(self, x):
        # TODO e: same tanh you've used everywhere else.
        #   out = torch.tanh(x)
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# -----------------------------------------------------------------------
class Embedding:
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, IX):
        # TODO f: the lookup -- exactly C[X] from mlp_train.py, just with
        # self.weight playing the role of C.
        #   out = self.weight[IX]
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


# -----------------------------------------------------------------------
class Flatten:
    def __call__(self, x):
        # TODO g: same flatten as emb.view(emb.shape[0], -1) in mlp_train.py.
        #   out = x.view(x.shape[0], -1)
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


# -----------------------------------------------------------------------
class FlattenConsecutive:
    """
    The actual WaveNet trick. Input x is (batch, time, channels) -- `time`
    is however many context positions are still un-merged at this point in
    the stack (starts at block_size, e.g. 8). Instead of flattening ALL of
    it like `Flatten` does, only fuse `n` NEIGHBORING time-steps into each
    other: (B, T, C) -> (B, T//n, C*n). Stack this behind a Linear+BN+Tanh,
    then do it again -- each round halves how many "positions" are left
    and doubles their channel width, e.g. for block_size=8, n=2:
        8 chars -> (Flatten2+Linear) -> 4 "pairs" -> (again) -> 2 "quads"
        -> (again) -> 1 "octet" -- now equivalent to the old flat Flatten,
        but reached through 3 shallow fusions instead of 1 giant one.
    This is exactly the tree/dilation structure from the WaveNet paper.
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        # TODO i: group every `n` consecutive time-steps into one, by
        # merging them into the channel dim: (B, T, C) -> (B, T//n, C*n).
        #   out = x.view(B, T // self.n, C * self.n)
        out = x.view(B, T // self.n, C * self.n)
        if out.shape[1] == 1:
            # once only ONE "position" is left, drop the now-useless time
            # dim entirely -- (B, 1, C*n) -> (B, C*n) -- so the next Linear
            # sees a plain 2D input, same shape Linear/BatchNorm1d already
            # expect from the non-hierarchical model.
            out = out.squeeze(1)
        self.out = out
        return self.out

    def parameters(self):
        return []


# -----------------------------------------------------------------------
class Sequential:
    """Chains layers: output of one feeds into the next, in order."""
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        # TODO j: put every BatchNorm1d (and any other layer that has a
        # `.training` attribute) into TRAINING mode -- i.e. loop over
        # self.layers and, for any layer that has a `.training` attribute,
        # set it to True. Layers without one (Linear, Tanh, ...) just get
        # skipped -- `hasattr(layer, "training")` tells you which is which.
        #   for layer in self.layers:
        #       if hasattr(layer, "training"):
        #           layer.training = True
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        # TODO k: same idea, but set .training = False -- so BatchNorm1d
        # switches to using its running_mean/running_var instead of the
        # current (dev/test) minibatch's stats.
        #   for layer in self.layers:
        #       if hasattr(layer, "training"):
        #           layer.training = False
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False

# =========================================================================
# self-test -- not real training, just sanity-checking shapes/wiring
if __name__ == "__main__":
    g = torch.Generator().manual_seed(42)
    n, fan_in, fan_out = 16, 10, 5

    model = Sequential([
        Linear(fan_in, fan_out, bias=False, generator=g),
        BatchNorm1d(fan_out),
        Tanh(),
    ])

    x = torch.randn((n, fan_in), generator=g)
    out = model(x)

    print("output shape:", tuple(out.shape), " (should be (16, 5))")
    print("num parameter tensors:", len(model.parameters()),
          " (should be 3: Linear.weight, BatchNorm gamma, BatchNorm beta)")
    print("output mean/std:", out.mean().item(), out.std().item(),
          " (should be roughly near 0 / well inside (-1,1), tanh+batchnorm keeps it tame)")

    # ---- part 2: the actual hierarchical WaveNet stack -----------------
    # fake "context" input: batch of 4 examples, block_size=8 characters
    # of context each, vocab of 27 (matching names.txt), n_embd=10.
    BATCH, BLOCK_SIZE, VOCAB, N_EMBD, N_HIDDEN = 4, 8, 27, 10, 68

    wavenet = Sequential([
        Embedding(VOCAB, N_EMBD, generator=g),                      # (B, 8, 10)
        FlattenConsecutive(2), Linear(N_EMBD * 2, N_HIDDEN, bias=False, generator=g), BatchNorm1d(N_HIDDEN), Tanh(),   # (B, 4, 68)
        FlattenConsecutive(2), Linear(N_HIDDEN * 2, N_HIDDEN, bias=False, generator=g), BatchNorm1d(N_HIDDEN), Tanh(), # (B, 2, 68)
        FlattenConsecutive(2), Linear(N_HIDDEN * 2, N_HIDDEN, bias=False, generator=g), BatchNorm1d(N_HIDDEN), Tanh(), # (B, 68)
        Linear(N_HIDDEN, VOCAB, generator=g),                        # (B, 27)
    ])

    Xb = torch.randint(0, VOCAB, (BATCH, BLOCK_SIZE), generator=g)
    logits = wavenet(Xb)
    for layer in wavenet.layers:
        print(layer.__class__.__name__,": ", tuple(layer.out.shape))
