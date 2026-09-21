import matplotlib.pyplot as plt
import seaborn           as sns
import torch
import torch.nn.functional as F

from utils.get_dataset import get_dataset
sns.set_theme(style="darkgrid", context="talk", font_scale=0.9)
gen        = torch.Generator().manual_seed(2147483647)

words = get_dataset()

print(f"Total vocabulary available is: {len(words)}")

chars: list[str] = ['.']  + sorted(list(set(''.join(words))))

lookup_table: dict[str, int] = {}
rev_lkp_tbl : dict[int, str] = {}

for idx, ch in enumerate(chars):
    lookup_table[ch] = idx
    rev_lkp_tbl[idx] = ch

block_size = 3                                                                      # The number of characters required to predict the next character, i.e, the context used to predict the next token
X, Y       = [], []
for w in words:
    context = [0] *  block_size                                                     # Create the context window padded with the index of the start token
    for ch in w + '.':
        idx = lookup_table[ch]                                                      # Look for the index of the current character
        X.append(context)                                                           # Context that should be used to predict the current character
        Y.append(idx)                                                               # Index of the current character that should be predicted
        context = context[1:] + [idx]                                               # Slide it forward by the block size and append the current character to the context window


X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 2), generator=gen)                                             # Random weights for the embedding layer

# Hidden layer 1
W1 = torch.randn((6, 100) , generator=gen)                                          # Embedding layer produces a 3 x 2 output, so we create a matrix with 6 rows and 100 columns ( for 100 neurons )
b1 = torch.randn(100      , generator=gen)                                          # 100 biases for 100 neurons
# Output Layer
W2 = torch.randn((100, 27), generator=gen)                                          # First hidden layer emits 100 outputs
b2 = torch.randn(27       , generator=gen)

parameters = [C, W1, b1, W2, b2]
print(f"Total parameters = {sum(p.nelement() for p in parameters)}")
for p in parameters:
    p.requires_grad = True

emb = C[X]
h1 = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h1 @ W2 + b2

loss = F.cross_entropy(logits, Y)
print(f"Initial loss on entire training set: {loss:.4f}")



exp_learning_rates = torch.linspace(start=-3, end=0, steps=1000)
learning_rates     = 10 ** exp_learning_rates

learning_rates_used = []
losses              = []


for i in range(1000):
    minibatch_indices = torch.randint(0, X.shape[0], (32, ))

    emb = C[X[minibatch_indices]]
    h1 = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h1 @ W2 + b2

    loss = F.cross_entropy(logits, Y[minibatch_indices])
    print(f"Cross Entropy Loss after {i + 1} iterations is: {loss:.4f}")

    for p in parameters:
        p.grad = None

    loss.backward()
    for p in parameters:
        p.data += -learning_rates[i] * p.grad

    losses.append(loss.item())
    learning_rates_used.append(exp_learning_rates[i])

plt.plot(learning_rates_used, losses)
plt.xlabel("Learning Rate Exponent")
plt.ylabel("Loss")
plt.savefig("learning_rate_estimation.jpeg")

# Reset Model parameters
C = torch.randn((27, 2)   , generator=gen)
W1 = torch.randn((6, 100) , generator=gen)
b1 = torch.randn(100      , generator=gen)
W2 = torch.randn((100, 27), generator=gen)
b2 = torch.randn(27       , generator=gen)

parameters = [C, W1, b1, W2, b2]
print(f"Total parameters = {sum(p.nelement() for p in parameters)}")
plt.figure(figsize=(12, 12))
for p in parameters:
    p.requires_grad = True
ax = sns.scatterplot(x=C[:, 0].data, y=C[:, 1].data,  s=500, color="#2b5b84", edgecolor="white",  linewidth=1.5, alpha=0.95)

for i in range(C.shape[0]):
    ax.text(C[i, 0].item(), C[i, 1].item(), rev_lkp_tbl[i], ha="center", va="center", color="white", fontsize=10, fontweight="bold")


ax.set_title("Text Embeddings Before Training", pad=20, fontweight="bold")
ax.set_xlabel("Latent Dimension 1", labelpad=15, color="#555555")
ax.set_ylabel("Latent Dimension 2", labelpad=15, color="#555555")

sns.despine(left=True, bottom=True)
plt.savefig("text_embeddings_before_training.jpeg", dpi=300, bbox_inches="tight")


for i in range(30000):
    minibatch_indices = torch.randint(0, X.shape[0], (32, ), generator=gen)

    emb = C[X[minibatch_indices]]
    h1 = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h1 @ W2 + b2

    loss = F.cross_entropy(logits, Y[minibatch_indices])
    print(f"Cross Entropy Loss after {i + 1} iterations is: {loss:.4f}")

    for p in parameters:
        p.grad = None

    loss.backward()
    for p in parameters:
        p.data += -0.01 * p.grad


emb    = C[X]
h1     = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h1 @ W2 + b2

loss = F.cross_entropy(logits, Y)
print(f"Final loss on entire training set: {loss:.4f}")
plt.figure(figsize=(12, 12))
for p in parameters:
    p.requires_grad = True
ax = sns.scatterplot(x=C[:, 0].data, y=C[:, 1].data,  s=500, color="#2b5b84", edgecolor="white",  linewidth=1.5, alpha=0.95)

for i in range(C.shape[0]):
    ax.text(C[i, 0].item(), C[i, 1].item(), rev_lkp_tbl[i], ha="center", va="center", color="white", fontsize=10, fontweight="bold")


ax.set_title("Text Embeddings After Training", pad=20, fontweight="bold")
ax.set_xlabel("Latent Dimension 1", labelpad=15, color="#555555")
ax.set_ylabel("Latent Dimension 2", labelpad=15, color="#555555")

sns.despine(left=True, bottom=True)
plt.savefig("text_embeddings_after_training.jpeg", dpi=300, bbox_inches="tight")

