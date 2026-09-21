"""
Character-Level MLP Language Model.

This script implements a character-level language model using a multi-character
context and a simple multilayer perceptron (MLP).

Instead of predicting the next character using only the immediately preceding
character, the model uses a fixed-size context window of previous characters.

The model:

* Builds training examples using a 3-character context window.
* Learns a low-dimensional embedding for each character.
* Flattens the character embeddings into a single input vector.
* Passes the input through a hidden layer with tanh activation.
* Predicts the next character from 27 possible characters.
* Uses cross-entropy loss for training.
* Trains the model using mini-batch gradient descent.
* Uses separate training, validation, and test datasets.
* Visualizes the learned character embeddings before and after training.
* Generates names by sampling from the model's predicted probabilities.

This implementation extends the earlier count-based and neural bigram models
toward a neural language model that can use multiple previous characters as
context.
"""

import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils.embedding_plotter import plot_embeddings
from utils.make_dataset      import build_dataset
from utils.get_dataset       import get_dataset
sns.set_theme(style="darkgrid", context="talk", font_scale=0.9)
random.seed(42)

gen        = torch.Generator().manual_seed(2147483647)

words = get_dataset()

print(f"Total vocabulary available is: {len(words)}")

chars: list[str] = ['.']  + sorted(list(set(''.join(words))))

lookup_table: dict[str, int] = {}
rev_lkp_tbl : dict[int, str] = {}

for idx, ch in enumerate(chars):
    lookup_table[ch] = idx
    rev_lkp_tbl[idx] = ch
block_size = 3

random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

X_train, Y_train = build_dataset(words[:n1]  , block_size, lookup_table)
X_val  , Y_val   = build_dataset(words[n1:n2], block_size, lookup_table)
X_test , Y_test  = build_dataset(words[n2:]  , block_size, lookup_table)

C  = torch.randn((27, 10) , generator=gen)   # Embedding matrix: 27 characters represented using 10 dimensions
W1 = torch.randn((30, 200), generator=gen)   # weights for the first hidden layer
b1 = torch.randn(200      , generator=gen)   # Biases for the 200 neurons in the first layer
W2 = torch.randn((200, 27), generator=gen)   # Weights for the output layer
b2 = torch.randn(27       , generator=gen)   # Bias for the 27 output neurons

params = [C, W1, b1, W2, b2]                 # Collecting the parameters to update during the training process
for p in params:
    p.requires_grad = True

plot_embeddings(C, rev_lkp_tbl, "before")

print(f"Total parameters is : {sum(p.nelement() for p in params)}")
steps: list[int]    = []
losses: list[float] = []

for i in range(200000):
    minibatch_indices = torch.randint(0, X_train.shape[0], (50, ), generator=gen)
    emb = C[X_train[minibatch_indices]]
    h1 = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h1 @ W2 + b2
    loss = F.cross_entropy(logits, Y_train[minibatch_indices])
    if i % 1000 == 0:
        print(f"Cross entropy loss after {i} iterations is {loss}")
    for p in params:
        p.grad = None

    loss.backward()
    lr = 0.01 if i < 1_00_000 else 0.1
    for p in params:
        p.data += -lr * p.grad

    steps.append(i)
    losses.append(loss.log10().item())

plt.figure(figsize=(12, 6))
ax = sns.lineplot( x=steps, y=losses, color="#c0392b", linewidth=2.5, alpha=0.9)
plt.fill_between(steps, losses, color="#c0392b", alpha=0.1)
ax.set_title("Training Loss Over Time", pad=20, fontweight="bold")
ax.set_xlabel("Training Steps", labelpad=15, color="#555555")
ax.set_ylabel("$\\log_{10}$(Loss)", labelpad=15, color="#555555")
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("training_losses_over_time.jpeg")

emb    = C[X_test]
h1     = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h1 @ W2 + b2

loss = F.cross_entropy(logits, Y_test)
print(f"Loss on test set: {loss:.4f}")

emb    = C[X_val]
h1     = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h1 @ W2 + b2

loss = F.cross_entropy(logits, Y_val)

print(f"Loss on Validation set: {loss:.4f}")

plot_embeddings(C, rev_lkp_tbl, "after")


# sampling
inf_gen = torch.Generator().manual_seed(2147483647 + 20)

for i in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([[context]])]
        h   = torch.tanh(emb.view(-1, 30) @ W1 + b1)
        logits = h @ W2 + b2
        probs  = F.softmax(logits, dim=1)
        idx    = torch.multinomial(probs, num_samples=1, replacement=True, generator=inf_gen).item()
        context = context[1:] + [idx]
        out.append(idx)
        if idx == 0:
            break

    print(''.join(rev_lkp_tbl[idx] for idx in out))
