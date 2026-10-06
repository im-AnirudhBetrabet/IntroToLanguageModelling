import random

from Bigrams.utils.get_dataset  import get_dataset
from Bigrams.utils.make_dataset import build_dataset
from nn.batch_norm_1d           import BatchNorm1d
from nn.Linear                  import Linear
from nn.TanH                    import Tanh
from nn.embedding               import Embedding
from nn.sequential              import Sequential
from nn.flatten_consecutive     import FlattenConsecutive

import torch
import matplotlib.pyplot   as plt
import seaborn             as sns
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)

print(">> Reading dataset.")
names = get_dataset()
print(">> Dataset read successfully.")

print(">> Generating vocabulary.")
vocabulary = ['.'] +  list(sorted(set(list(''.join(names)))))
print(f">> Vocabulary generated. Unique characters available in dataset is: {len(vocabulary)}.")

_CONTEXT_WINDOW      = 8
_VOCABULARY_SIZE     = len(vocabulary)
_EMBEDDING_DIMENSION = 24
_HIDDEN_NEURONS      = 128
_MINI_BATCH_SIZE     = 32
_ITERATIONS          = 200000

char_lookup_table: dict[str, int] = {}
rev_lookup_table : dict[int, str] = {}

print(">> Generating lookup tables.")
for i, ch in enumerate(vocabulary):
    char_lookup_table[ch] = i
    rev_lookup_table[i]   = ch
print(">> Lookup tables generated.")

random.shuffle(names)
n1: int = int(0.8 * len(names))
n2: int = int(0.9 * len(names))

X_train, Y_train = build_dataset(names[  :n1], _CONTEXT_WINDOW, char_lookup_table)
X_val  , Y_val   = build_dataset(names[n1:n2], _CONTEXT_WINDOW, char_lookup_table)
X_test , Y_test  = build_dataset(names[n2:  ], _CONTEXT_WINDOW, char_lookup_table)


model = Sequential([
    Embedding(_VOCABULARY_SIZE , _EMBEDDING_DIMENSION),
    FlattenConsecutive(2), Linear(_EMBEDDING_DIMENSION * 2, _HIDDEN_NEURONS, bias=False), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    FlattenConsecutive(2), Linear(_HIDDEN_NEURONS * 2     , _HIDDEN_NEURONS, bias=False), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    FlattenConsecutive(2), Linear(_HIDDEN_NEURONS * 2     , _HIDDEN_NEURONS, bias=False), BatchNorm1d(_HIDDEN_NEURONS), Tanh(),
    Linear(_HIDDEN_NEURONS , _VOCABULARY_SIZE)
])

with torch.no_grad():
  model._layers[-1]._weights *= 0.1

parameters = model.parameters()

print(f">> Model initialized with {sum(p.nelement() for p in parameters)} parameters.")
for p in parameters:
  p.requires_grad = True

losses = []

print(f">> Starting training...")
for i in range(_ITERATIONS):
    ix = torch.randint(0, X_train.shape[0], (_MINI_BATCH_SIZE, ))

    X_batch, Y_batch = X_train[ix], Y_train[ix]
    logits = model(X_batch)

    loss = F.cross_entropy(logits, Y_batch)

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 100000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f">>> Training loss after {i} iterations is {loss.item():.4f}")

    losses.append(loss.item())
print(f">> Training completed.")
for layer in model._layers:
    print(f"{layer.__class__.__name__} : {tuple(layer.out.shape)}")

for layer in model._layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

@torch.no_grad()
def split_loss(split):
  x,y = {
    'train': (X_train, Y_train),
    'val'  : (X_val, Y_val),
    'test' : (X_test, Y_test),
  }[split]
  logits = model(x)
  loss   = F.cross_entropy(logits, y)
  print(f"{split} set loss is: {loss.item():.7f}")

split_loss('train')
split_loss('val')
split_loss('test')

plt.figure(figsize=(18, 5))

with torch.no_grad():
    loss_tensor = torch.tensor(losses).view(-1, 1000).mean(dim=1)
    y_vals      = loss_tensor.detach().cpu().numpy()
    x_vals      = range(len(y_vals))

    color = "#B2BD7E"
    sns.lineplot(x=x_vals, y=y_vals, linewidth=3, color=color, label="Training Loss (Rolling Mean)")
    plt.fill_between(x_vals, y_vals, alpha=0.12, color=color)
    plt.title('Model Convergence: Training Loss Progress', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Training Epoch / Iteration Chunk (x1000 Steps)', fontsize=13, labelpad=10)
    plt.ylabel('Mean Loss', fontsize=13, labelpad=10)
    plt.ylim(y_vals.min() * 0.9, y_vals.max() * 1.05)
    plt.xlim(0, len(y_vals) - 1)
    plt.legend(frameon=True, shadow=True, loc='upper right', fontsize=11)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("model_convergence.jpeg")

for _ in range(20):

    out = []
    context = [0] * _CONTEXT_WINDOW
    while True:
        logits  = model(torch.tensor([context]))
        probs   = F.softmax(logits, dim=1)
        ix      = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(rev_lookup_table[i] for i in out))