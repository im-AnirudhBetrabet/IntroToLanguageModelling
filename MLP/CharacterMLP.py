from Bigrams.utils.make_dataset      import build_dataset
from Bigrams.utils.get_dataset       import get_dataset
import torch
import torch.nn.functional as F
import random


random.seed(42)
gen  = torch.Generator().manual_seed(2147483647)
data = get_dataset()

print(f"Total words available are: {len(data)}")

vocab: list[str] = ['.'] + sorted(list(set(''.join(data))))

lookup_table: dict[str, int] = {}
rev_lkp_tbl : dict[int, str] = {}

for idx, ch in enumerate(vocab):
    lookup_table[ch] = idx
    rev_lkp_tbl[idx] = ch


_EMBEDDING_DIMENSION = 10
_CONTEXT_WINDOW      =  3
_VOCABULARY_SIZE     = len(vocab)
_HIDDEN_NEURONS      = 200

random.shuffle(data)

n1 = int(0.8 * len(data))
n2 = int(0.9 * len(data))

X_train, Y_train = build_dataset(data[:n1]  , _CONTEXT_WINDOW, lookup_table)
X_val  , Y_val   = build_dataset(data[n1:n2], _CONTEXT_WINDOW, lookup_table)
X_test , Y_test  = build_dataset(data[n2:]  , _CONTEXT_WINDOW, lookup_table)

embedding_weights = torch.randn((_VOCABULARY_SIZE, _EMBEDDING_DIMENSION)                 , generator=gen)  # Random weights for the embedding layer
h_layer_weights   = torch.randn((_EMBEDDING_DIMENSION * _CONTEXT_WINDOW, _HIDDEN_NEURONS), generator=gen)  # Random weights for the neurons in the hidden layer
h_layer_biases    = torch.randn(_HIDDEN_NEURONS                                          , generator=gen)  # Random biases for the neurons in the hidden layer.
op_layer_weights  = torch.randn((_HIDDEN_NEURONS, _VOCABULARY_SIZE)                      , generator=gen)  # Random weights for the neurons in the output layer.
op_layer_biases   = torch.randn(_VOCABULARY_SIZE                                         , generator=gen)  # Random biases for the neurons in the output layer.

available_params = [embedding_weights, h_layer_weights, h_layer_biases, op_layer_weights, op_layer_biases] # Collecting available parameters so that they can be updated during training.
print(f"Available parameters are: {sum(p.nelement() for p in available_params)}")


for p in available_params:
    p.requires_grad = True

for i in range(200000):
    minibatch_indices = torch.randint(0, X_train.shape[0], (50, ), generator=gen)
    emb = embedding_weights[X_train[minibatch_indices]]
    h1 = torch.tanh(emb.view(-1, _CONTEXT_WINDOW * _EMBEDDING_DIMENSION) @ h_layer_weights + h_layer_biases)
    logits = h1 @ op_layer_weights + op_layer_biases
    loss = F.cross_entropy(logits, Y_train[minibatch_indices])
    if i % 2000 == 0:
        print(f"Cross entropy loss after {i} iterations is {loss:.4f}")
    for p in available_params:
        p.grad = None

    loss.backward()
    lr = 0.01 if i < 1_00_000 else 0.1
    for p in available_params:
        p.data += -lr * p.grad

emb    = embedding_weights[X_test]
h1     = torch.tanh(emb.view(-1, _CONTEXT_WINDOW * _EMBEDDING_DIMENSION) @ h_layer_weights + h_layer_biases)
logits = h1 @ op_layer_weights + op_layer_biases

loss = F.cross_entropy(logits, Y_test)
print(f"Loss on test set: {loss:.4f}")

emb    = embedding_weights[X_val]
h1     = torch.tanh(emb.view(-1, _CONTEXT_WINDOW * _EMBEDDING_DIMENSION) @ h_layer_weights + h_layer_biases)
logits = h1 @ op_layer_weights + op_layer_biases

loss = F.cross_entropy(logits, Y_val)

print(f"Loss on Validation set: {loss:.4f}")


inf_gen = torch.Generator().manual_seed(2147483647 + 20)

for i in range(20):
    out = []
    context = [0] * _CONTEXT_WINDOW
    while True:
        emb = embedding_weights[torch.tensor([context])]
        h   = torch.tanh(emb.view(-1, _CONTEXT_WINDOW * _EMBEDDING_DIMENSION) @ h_layer_weights + h_layer_biases)
        logits = h @ op_layer_weights + op_layer_biases
        probs  = F.softmax(logits, dim=1)
        idx    = torch.multinomial(probs, num_samples=1, replacement=True, generator=inf_gen).item()
        context = context[1:] + [idx]
        out.append(idx)
        if idx == 0:
            break

    print(''.join(rev_lkp_tbl[idx] for idx in out))