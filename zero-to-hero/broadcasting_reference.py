import torch

m = torch.tensor([[1.,2.,3.],
                 [4.,5.,6.],
                 [7.,8.,9.]]) # (3, 3) tensor

wrong = m / m.sum(dim=1) # (3, 3) / (3,) tensor
right = m / m.sum(dim=1, keepdim=True) # (3, 3) / (3, 1) tensor

print(wrong.sum(1)) # tensor([1.0000, 1.0000, 1.0000])
print(right.sum(1)) # tensor([[1.0000], [1.0000], [1.0000]])