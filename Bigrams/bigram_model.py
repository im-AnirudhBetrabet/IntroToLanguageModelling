"""
Character-level Neural Bigram Language Model.

This script implements a neural-network-based bigram language model for the
names dataset.

It:
- Represents characters using one-hot encoding.
- Uses a weight matrix to learn the relationship between consecutive
  characters.
- Converts the model's raw outputs (logits) into probabilities using softmax.
- Evaluates the model using negative log likelihood with L2 regularization.
- Trains the weight matrix using gradient descent.
- Generates new names by sampling from the learned probability distribution.

This implementation demonstrates how the count-based bigram model can be
recreated using a simple neural network and gradient-based optimization.
"""

import torch
import torch.nn.functional as F

from Bigrams.utils.get_dataset import get_dataset

N = torch.zeros(size=[27, 27], dtype=torch.int32)

names = get_dataset()

x, y = [], []
vocabulary  : list[str]      = sorted(list(set(list(''.join(names)))))
lookup_table: dict[str, int] = {}
rev_lkp_tbl : dict[int, str] = {}
for i, s in enumerate(vocabulary):
    lookup_table[s] = i+1
    rev_lkp_tbl[i+1]= s

lookup_table['.'] = 0
rev_lkp_tbl[0] = '.'


# Creating the dataset from the names
for n in names:
    w = ['.'] + list(n) + ['.']
    for w1, w2 in zip(w, w[1:]):
        idx1 = lookup_table[w1]
        idx2 = lookup_table[w2]
        x.append(idx1)
        y.append(idx2)

X = torch.tensor(x)
Y = torch.tensor(y)

train_gen = torch.Generator().manual_seed(2147483647)
num = X.nelement()
W = torch.randn(size=(27, 27), generator=train_gen, requires_grad=True)
print(f"Number of training samples is: {num}")

X_Encoded = F.one_hot(X, num_classes=27).float()                # Encode the input variables

for i in range(200):
    logits = X_Encoded @ W                                      # Raw scores before softmax
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)                # Softmax
    loss = -probs[torch.arange(num), Y].log().mean() + 0.01 * (W**2).mean()
    print(f"Loss after {i + 1} iterations is {loss}")
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

inf_gen = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix  = 0
    while True:
        x_in = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = x_in @ W
        counts = logits.exp()
        p      = counts / counts.sum(1,keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=inf_gen).item()
        out.append(rev_lkp_tbl[ix])
        if ix == 0:
            break
    print("".join(out))