from nn.Linear                  import Linear
from nn.batch_norm_1d           import BatchNorm1d
from nn.TanH                    import Tanh
from Bigrams.utils.get_dataset  import get_dataset
from Bigrams.utils.make_dataset import build_dataset

import torch.nn.functional as F
import random
import torch
import matplotlib.pyplot as plt
import seaborn           as sns
random.seed(42)

dataset = get_dataset()
gen     = torch.Generator().manual_seed(2147483647)

print(f"Total words available in the dataset is {len(dataset)}")

vocab: list[str] = ['.'] + list(sorted(list(set("".join(dataset)))))

print(f"Total vocabulary available for training is {len(vocab)}")

lookup_table   : dict[str, int] = {}
reverse_lkp_tbl: dict[int, str] = {}

for i, ch in enumerate(vocab):
    lookup_table[ch]   = i
    reverse_lkp_tbl[i] = ch

_EMBEDDING_DIMENSION = 10
_CONTEXT_WINDOW      =  3
_VOCABULARY_SIZE     = len(vocab)
_HIDDEN_NEURONS      = 100
_MINI_BATCH_SIZE     = 32
_ITERATIONS          = 200000
random.shuffle(dataset)

n1 = int(0.8 * len(dataset))
n2 = int(0.9 * len(dataset))

X_train, Y_train = build_dataset(dataset[:n1]  , _CONTEXT_WINDOW, lookup_table)
X_val  , Y_val   = build_dataset(dataset[n1:n2], _CONTEXT_WINDOW, lookup_table)
X_test , Y_test  = build_dataset(dataset[n2:]  , _CONTEXT_WINDOW, lookup_table)


C = torch.randn((_VOCABULARY_SIZE, _EMBEDDING_DIMENSION), generator=gen)

layers = [
    Linear((_EMBEDDING_DIMENSION * _CONTEXT_WINDOW), _HIDDEN_NEURONS), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    Linear(_HIDDEN_NEURONS                         , _HIDDEN_NEURONS), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    Linear(_HIDDEN_NEURONS                         , _HIDDEN_NEURONS), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    Linear(_HIDDEN_NEURONS                         , _HIDDEN_NEURONS), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    Linear(_HIDDEN_NEURONS                         , _HIDDEN_NEURONS), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    Linear(_HIDDEN_NEURONS                         , _VOCABULARY_SIZE)
]

with torch.no_grad():
    layers[-1]._weights *= 0.1              # making the last layer less confident
    for layer in layers:
        if isinstance(layer, Linear):
            layer._weights *= 5/3           # applying a gain to all the linear layers


params = [C] + [p for layer in layers for p in layer.parameters()]
print(f"Total parameters available are: {sum(p.nelement() for p in params)}")
for p in params:
    p.requires_grad = True
losses = []

for i in range(_ITERATIONS):
    # constructing the mini batch
    ix = torch.randint(0, X_train.shape[0], (_MINI_BATCH_SIZE, ), generator=gen)
    X_batch , Y_batch = X_train[ix], Y_train[ix]

    embs = C[X_batch]
    x    = embs.view(embs.shape[0], -1)
    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, Y_batch)

    for layer in layers:
        layer.out.retain_grad()

    for p in params:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 100000 else 0.01

    for p in params:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f"{i:7d} / {_ITERATIONS:7d} : {loss.item():.4f}")
    losses.append(loss.log10().item())

sns.set_theme(style="darkgrid", context="talk", font_scale=0.9)

plt.figure(figsize=(20, 5))
tanh_layers = [l for l in layers[:-1] if isinstance(l, Tanh)]
colors      = sns.color_palette("husl", n_colors=len(tanh_layers))

for i, layer in enumerate(tanh_layers):
    t = layer.out

    # Print statistics
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (
        i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))

    hy, hx = torch.histogram(t, density=True)

    x_vals = hx[:-1].detach().cpu().numpy()
    y_vals = hy.detach().cpu().numpy()

    sns.lineplot(x=x_vals, y=y_vals, linewidth=1.25,
                 label=f'Layer {i} ({layer.__class__.__name__})', color=colors[i])
    plt.fill_between(x_vals, y_vals, alpha=0.15, color=colors[i])

plt.title('Tanh Activation Distribution Across Layers', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Activation Value', fontsize=14, labelpad=10)
plt.ylabel('Density', fontsize=14, labelpad=10)

plt.xlim(-1.05, 1.05)
plt.legend(title="Network Layers", frameon=True, shadow=True, loc='upper right')
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.savefig("activation_distributions.jpeg")

plt.figure(figsize=(20, 5))
colors = sns.color_palette("viridis", n_colors=len(tanh_layers))

for i, layer in enumerate(tanh_layers):
    t = layer.out.grad

    print('layer %d (%10s): mean %+f, std %e' % (
        i, layer.__class__.__name__, t.mean(), t.std()))

    hy, hx = torch.histogram(t, density=True)

    x_vals = hx[:-1].detach().cpu().numpy()
    y_vals = hy.detach().cpu().numpy()

    sns.lineplot(x=x_vals, y=y_vals, linewidth=1.25,
                 label=f'Layer {i} ({layer.__class__.__name__})', color=colors[i])
    plt.fill_between(x_vals, y_vals, alpha=0.15, color=colors[i])

plt.title('Gradient Distribution Across Tanh Layers', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Gradient Value', fontsize=14, labelpad=10)
plt.ylabel('Density', fontsize=14, labelpad=10)

# Clean up the legend and axes
plt.legend(title="Network Layers", frameon=True, shadow=True, loc='upper right')
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.savefig("gradient_distribution.jpeg")

plt.figure(figsize=(20, 5))
weights_2d = [p for p in params if p.ndim == 2]
colors = sns.color_palette("magma", n_colors=len(weights_2d))

color_idx = 0
for i, p in enumerate(params):
    t = p.grad
    if p.ndim == 2:
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (
            tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))

        hy, hx = torch.histogram(t, density=True)

        x_vals = hx[:-1].detach().cpu().numpy()
        y_vals = hy.detach().cpu().numpy()

        sns.lineplot(x=x_vals, y=y_vals, linewidth=1.25,
                     label=f'Param {i} {tuple(p.shape)}', color=colors[color_idx])
        plt.fill_between(x_vals, y_vals, alpha=0.15, color=colors[color_idx])

        color_idx += 1

plt.title('Weight Gradients Distribution (2D Parameters)', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Gradient Value', fontsize=14, labelpad=10)
plt.ylabel('Density', fontsize=14, labelpad=10)
plt.legend(title="Weight Dimensions", frameon=True, shadow=True, loc='upper right')
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.savefig("weight_gradient_distribution.jpeg")