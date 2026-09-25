import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

chars = sorted(list(set(input_text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(input_text), dtype=torch.long)
n=int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # context length for predictions
eval_interval = 500 # how often (in training steps) to run a proper loss evaluation
eval_iters = 200 # how many batches to average over when evaluating

def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1].tolist()
        target = yb[b, t].tolist()

torch.manual_seed(1337)

n_embd = 32  # size of the token/position embedding vectors (C, before any attention head)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        # tril is not a parameter -- it's never trained, so register it as a buffer
        # (still moves with .to(device) and gets saved/loaded, but no gradient)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size) -- "what do I contain"
        q = self.query(x) # (B, T, head_size) -- "what am I looking for"

        # TODO 1: compute attention affinities ("wei") via a scaled dot product of q and k.
        #   wei[b, t, s] should measure how much token t attends to token s.
        #   Shapes: q is (B,T,head_size), k is (B,T,head_size) -- you need k transposed
        #   on its last two dims (k.transpose(-2, -1)) so the matmul gives (B,T,T).
        #   Don't forget the * head_size**-0.5 scaling discussed above.
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # TODO 2: apply the causal mask so token t can't attend to tokens s > t.
        #   Use self.tril[:T, :T] == 0 as the condition, and wei.masked_fill(condition, float('-inf')).
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # your code here

        # TODO 3: turn the masked affinities into a proper probability distribution per row.
        #   F.softmax(..., dim=-1)  -- dim=-1 because each ROW (fixed t, varying s) should sum to 1.
        wei = F.softmax(wei, dim=-1)  # your code here

        # TODO 4: compute the value vectors and aggregate them using wei.
        v = self.value(x)       # your code here -- same shape pattern as k and q above
        out = wei @ v     # your code here -- (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # TODO: create num_heads separate Head(head_size) instances, wrapped in an
        #   nn.ModuleList (NOT a plain Python list -- nn.ModuleList is what makes
        #   PyTorch register each Head's parameters so they show up in .parameters(),
        #   get moved by .to(device), get saved by state_dict(), etc. A plain list is
        #   invisible to all of that bookkeeping, even though iterating it works fine).
        #   A list comprehension inside nn.ModuleList([...]) is the idiomatic way.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # your code here

    def forward(self, x):
        # TODO: run x through every head in self.heads, then concatenate their outputs
        #   along the last dimension (dim=-1). Each head outputs (B, T, head_size);
        #   concatenating num_heads of them gives (B, T, head_size * num_heads).
        #   Hint: a list comprehension [h(x) for h in self.heads] + torch.cat(..., dim=-1)
        return torch.cat([h(x) for h in self.heads], dim=-1).view(x.shape[0], x.shape[1], -1)  # your code here


class FeedForward(nn.Module):
    """ a simple per-token linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        # TODO: build self.net as an nn.Sequential of:
        #   nn.Linear(n_embd, n_embd), then nn.ReLU()
        #   Without the ReLU, stacking this Linear with lm_head right after would
        #   collapse into one big linear transform -- no added expressive power.
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: run x through self.net. Note this Linear only acts on the last
        #   dimension (n_embd) -- PyTorch applies it independently at every (B, T)
        #   position, so no information moves between tokens here (unlike attention).
        return self.net(x)  # your code here


class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (feedforward),
        each wrapped in a residual connection and preceded by LayerNorm. """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # TODO: create two separate LayerNorm instances, ln1 and ln2, each via
        #   nn.LayerNorm(n_embd). Two separate instances because each has its own
        #   learnable scale/shift parameters -- they must not be shared between the
        #   attention sub-layer and the feedforward sub-layer.
        self.ln1 = nn.LayerNorm(n_embd)  # your code here
        self.ln2 = nn.LayerNorm(n_embd)  # your code here

    def forward(self, x):
        # TODO: residual connection around attention: x = x + self.sa(self.ln1(x))
        #   Note LayerNorm is applied BEFORE self.sa, not after -- this is "pre-norm",
        #   the modern standard (normalize the input to a sub-layer, not its output).
        x = x + self.sa(self.ln1(x))  # your code here
        # TODO: same pattern for feedforward: x = x + self.ffwd(self.ln2(x))
        x = x + self.ffwd(self.ln2(x))  # your code here
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # TODO: add a position embedding table -- one learned vector per position
        # 0..block_size-1 (nn.Embedding(num_embeddings, embedding_dim), same idea as
        # token_embedding_table but indexed by position instead of by token identity).
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # your code here
        n_head = 4  # 4 heads of n_embd // n_head each, concatenated back to n_embd
        n_layer = 3  # how many Blocks to stack
        # TODO: build self.blocks as an nn.Sequential of n_layer Block(n_embd, n_head)
        #   instances. nn.Sequential works here (unlike the ModuleList in
        #   MultiHeadAttention) because each Block just takes x and returns a
        #   same-shaped x -- a plain feed-forward chain, exactly what Sequential
        #   auto-chains for you. A list comprehension inside nn.Sequential([...]) works,
        #   same idiom as the ModuleList case.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])  # your code here
        # TODO: one final nn.LayerNorm(n_embd), applied after all the blocks and before
        #   lm_head -- standard practice to normalize right before the final projection.
        self.ln_f = nn.LayerNorm(n_embd)  # your code here
        # TODO: add lm_head -- a linear layer mapping n_embd -> vocab_size. This produces
        # the actual logits over the vocabulary from the attention output.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # your code here

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # TODO: compute pos_emb by calling self.position_embedding_table on torch.arange(T)
        #   (a 1D tensor [0, 1, ..., T-1]). Result shape: (T, n_embd) -- it broadcasts
        #   against tok_emb's (B, T, n_embd) when you add them below.
        pos_emb = self.position_embedding_table(torch.arange(T))  # your code here
        # TODO: combine token identity and position into one input (simple elementwise sum)
        x = tok_emb + pos_emb  # your code here -- (B, T, n_embd)
        # TODO: run x through the stacked blocks
        x = self.blocks(x)  # your code here -- (B, T, n_embd)
        # TODO: apply the final LayerNorm
        x = self.ln_f(x)  # your code here -- (B, T, n_embd)
        # TODO: project through lm_head to get logits over the vocabulary
        logits = self.lm_head(x)  # your code here -- (B, T, vocab_size)
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # TODO: crop idx to the last block_size tokens before feeding it in --
            #   position_embedding_table only has entries for 0..block_size-1, so feeding
            #   more than block_size tokens would index past the end of that table.
            idx_cond = idx[:, -block_size:]  # your code here
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)

# @torch.no_grad() tells autograd not to track any operations inside this function --
# we're only reading the loss here, never calling .backward() on it, so building the
# computation graph would just waste memory and compute for nothing.
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()  # switch to eval mode (no effect yet with no Dropout, but correct practice)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # TODO: sample a batch from `split` (get_batch(split)) and run the model
            #   on it to get logits and loss (same pattern as the training loop below).
            #   Store the scalar into losses[k] (loss.item()).
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()  # your code here

    # evaluate the loss
        # TODO: average the eval_iters loss samples for this split and store in out[split]
        out[split] = losses.mean()  # your code here
    m.train()  # switch back to train mode before returning to the training loop
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for i in range(10000):
    # TODO: every eval_interval steps (and on the very last step), call estimate_loss()
    #   and print something like f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    #   Hint: if i % eval_interval == 0 or i == 9999:
    if i % eval_interval == 0 or i == 9999:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))