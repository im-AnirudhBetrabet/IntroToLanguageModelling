from Bigrams.utils.get_dataset  import get_dataset
from Bigrams.utils.make_dataset import build_dataset

import torch
import torch.nn.functional as F
import matplotlib.pyplot   as plt

# utility function to compare the gradients calculated manually against those determined by pytorch
def compare(s, dt, t):
    ex  = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    max_diff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):s} | max difference: {max_diff}')


# retrieving the names dataset
names = get_dataset()

# creating the vocabulary , i.e., the unique characters that occur in the dataset
vocabulary = ['.'] + list(sorted(list(set(''.join(names)))))

# creating the lookup and reverse lookup table
lkp_table    : dict[str, int] = {}
rev_lkl_table: dict[int, str] = {}


for i, ch in enumerate(vocabulary):
    lkp_table[ch]    = i
    rev_lkl_table[i] = ch

# constants for training
_EMBEDDING_DIMENSION = 10
_CONTEXT_WINDOW      =  3
_VOCABULARY_SIZE     = len(vocabulary)
_HIDDEN_NEURONS      = 100
_MINI_BATCH_SIZE     = 32
_ITERATIONS          = 200000


n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

X_train, Y_train = build_dataset(names[  :n1], _CONTEXT_WINDOW, lkp_table)
X_val  , Y_val   = build_dataset(names[n1:n2], _CONTEXT_WINDOW, lkp_table)
X_test , Y_test  = build_dataset(names[n2:  ], _CONTEXT_WINDOW, lkp_table)

gen = torch.Generator().manual_seed(2147483647)

# ----------------- EMBEDDINGS -----------------
C = torch.randn((_VOCABULARY_SIZE, _EMBEDDING_DIMENSION)                  , generator=gen)

# ----------------- LAYER 1 -----------------
W1 = torch.randn((_EMBEDDING_DIMENSION * _CONTEXT_WINDOW, _HIDDEN_NEURONS), generator=gen) * ( 5 / 3) / ((_EMBEDDING_DIMENSION * _CONTEXT_WINDOW) ** 0.5)
b1 = torch.randn(_HIDDEN_NEURONS                                          , generator=gen) * 0.1

# ----------------- LAYER 2 -----------------
W2 = torch.randn((_HIDDEN_NEURONS, _VOCABULARY_SIZE)                      , generator=gen) * 0.1
b2 = torch.randn(_VOCABULARY_SIZE                                         , generator=gen) * 0.1

# ----------------- BATCH NORM PARAMS -----------------
bn_gain = torch.randn((1, _HIDDEN_NEURONS)) * 0.1 + 1.0
bn_bias = torch.randn((1, _HIDDEN_NEURONS)) * 0.1

params = [C, W1, b1, W2, b2, bn_gain, bn_bias]
print(f"Total parameters in the model are: {sum(p.nelement() for p in params)}")
for p in params:
    p.requires_grad = True

# ----------------- Get random indices for mini-batch training -----------------
batch_indexes    = torch.randint(0, X_train.shape[0], (_MINI_BATCH_SIZE, ), generator=gen)
X_batch, Y_batch = X_train[batch_indexes], Y_train[batch_indexes]


# ----------------- Create Embeddings -----------------
emb      = C[X_batch]
emb_trns = emb.view(emb.shape[0], -1)

# ----------------- Linear Layer 1 -----------------
h_pre_bnorm  = emb_trns @ W1 + b1                                               # Pre activation output for hidden layer 1
curr_bn_mean = (1 / _MINI_BATCH_SIZE) * h_pre_bnorm.sum(0, keepdim=True)        # Current batch mean
b_norm_diff  = h_pre_bnorm - curr_bn_mean                                       # Difference from the batch mean
b_norm_diff2 = b_norm_diff ** 2                                                 # Squared deviations used to calculate variance
b_norm_var   = (1 / (_MINI_BATCH_SIZE - 1)) * b_norm_diff2.sum(0, keepdim=True) # Calculating the batch norm variance

b_norm_var_inv = (b_norm_var + 1e-5) ** -0.5
batch_norm     = b_norm_diff * b_norm_var_inv                                   # Batch normalization ( ( x - mean ) / std )

h_pre_act = bn_gain * batch_norm + bn_bias                                      # Scaling and shifting

h = torch.tanh(h_pre_act)                                                       # Non-linear activating

# ----------------- Linear Layer 2 -----------------
logits = h @ W2 + b2

# ----------------- Softmax + Cross entropy Loss -----------------
logit_maxes = logits.max(1, keepdim=True).values                                # Calculating the maximum value of the logits
norm_logits = logits - logit_maxes                                              # Subtracting the maximum logit value from all values for numerical stability

counts = norm_logits.exp()

counts_sum     = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs          = counts * counts_sum_inv                                        # Calculating the softmax
log_probs      = probs.log()
loss           = -log_probs[range(_MINI_BATCH_SIZE), Y_batch].mean()            # Negative log likelihood / cross entropy loss

for p in params:
    p.grad = None

for t in [log_probs  , probs      , counts_sum_inv, counts_sum , counts        , norm_logits, logit_maxes,
          logits     , h          , h_pre_act     , batch_norm , b_norm_var_inv, b_norm_var , b_norm_diff2,
          b_norm_diff, h_pre_bnorm, curr_bn_mean  , emb_trns   , emb]:
    t.retain_grad()
loss.backward()
print(loss)

# ----------------- Manual back propagation -----------------
## 1. gradient of loss w.r.t log probability
grad_log_probs = torch.zeros_like(log_probs)
grad_log_probs[range(_MINI_BATCH_SIZE), Y_batch] = -1.0 / _MINI_BATCH_SIZE
compare('logprobs', grad_log_probs, log_probs)

## 2. gradient of loss w.r.t probs
grad_probs = grad_log_probs *  (1.0 / probs)
compare('probs', grad_probs, probs)

## 3. gradient of loss w.r.t count_sum_inv
grad_count_sum_inv = (grad_probs * counts).sum(1, keepdim=True)
compare('count_sum_inv', grad_count_sum_inv, counts_sum_inv)


## 4. gradient of loss w.r.t count_sum
grad_count_sum = (-counts_sum ** -2) * grad_count_sum_inv
compare('count_sum', grad_count_sum, counts_sum)

## 5. gradient of loss w.r.t counts
grad_counts  = counts_sum_inv * grad_probs                  # branch 2 where counts is used to determine probs
grad_counts += torch.ones_like(counts) * grad_count_sum     # branch 1 where counts contributes to counts_sum
compare('counts', grad_counts, counts)

# 6. gradient of loss w.r.t normalized logits
grad_norm_logits = grad_counts * norm_logits.exp()
compare('norm_logits', grad_norm_logits, norm_logits)

# 7. gradient of loss w.r.t logit_maxes
grad_logit_maxes = (-grad_norm_logits).sum(1, keepdim=True)
compare('logit_maxes', grad_logit_maxes, logit_maxes)

# 8. gradient of loss w.r.t logits
grad_logits  = grad_norm_logits.clone()                                                         # branch 2 where logits is used to determine the normalized logits
grad_logits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * grad_logit_maxes # branch 1 where logits is used to determine the maximum logits
compare('logits', grad_logits, logits)

# 9. gradient of loss w.r.t h
grad_h = grad_logits @ W2.T
compare('h', grad_h, h)

# 10. gradient of loss w.r.t W2
grad_W2 = h.T @ grad_logits
compare('W2', grad_W2, W2)

# 11. gradient of loss w.r.t B2
grad_B2 = grad_logits.sum(0)
compare('B2', grad_B2, b2)

# 12. gradient of loss w.r.t h_pre_act
grad_h_pre_act = (1.0 - h ** 2) * grad_h
compare('h_pre_act', grad_h_pre_act, h_pre_act)

# 13. gradient of loss w.r.t bn_gain
grad_bn_gain = (grad_h_pre_act * batch_norm).sum(0, keepdim=True)
compare('bn_gain', grad_bn_gain, bn_gain)

# 14. gradient of loss w.r.t batch_norm
grad_batch_norm = bn_gain * grad_h_pre_act
compare('batch_norm', grad_batch_norm, batch_norm)

# 15. gradient of loss w.r.t bn_bias
grad_bn_bias = grad_h_pre_act.sum(0, keepdim=True)
compare('bn_bias', grad_bn_bias, bn_bias)

# 16. gradient of loss w.r.t b_norm_var_inv
grad_b_norm_var_inv = (grad_batch_norm * b_norm_diff).sum(0, keepdim=True)
compare('b_norm_var_inv', grad_b_norm_var_inv, b_norm_var_inv)

# 17. gradient of loss w.r.t b_norm_var
grad_b_norm_var = (grad_b_norm_var_inv * (- 0.5 * (b_norm_var + 1e-5) ** -1.5))
compare('b_norm_var', grad_b_norm_var, b_norm_var)

# 18. gradient of loss w.r.t b_norm_diff2
grad_b_norm_diff2 = grad_b_norm_var * (1.0 / ( _MINI_BATCH_SIZE - 1) * torch.ones_like(b_norm_diff2))
compare('b_norm_diff2', grad_b_norm_diff2, b_norm_diff2)

# 19. gradient of loss w.r.t b_norm_diff
grad_b_norm_diff  = grad_batch_norm * b_norm_var_inv
grad_b_norm_diff += 2 * b_norm_diff * grad_b_norm_diff2
compare('b_norm_diff', grad_b_norm_diff, b_norm_diff)

# 20. gradient of loss w.r.t curr_bn_mean
grad_curr_bn_mean = (-grad_b_norm_diff).sum(0, keepdim=True)
compare('curr_bn_mean', grad_curr_bn_mean, curr_bn_mean)

# 21. gradient of loss w.r.t h_pre_bnorm
grad_h_pre_bnorm  = grad_b_norm_diff.clone()
grad_h_pre_bnorm += (1.0 / _MINI_BATCH_SIZE) * torch.ones_like(h_pre_bnorm) * grad_curr_bn_mean
compare('h_pre_bnorm', grad_h_pre_bnorm, h_pre_bnorm)

# 22. gradient of loss w.r.t W1
grad_W1 = emb_trns.T @ grad_h_pre_bnorm
compare('W1', grad_W1, W1)

# 23. gradient of loss w.r.t b1
grad_B1 = grad_h_pre_bnorm.sum(0)
compare('B1', grad_B1, b1)

# 24. gradient of loss w.r.t emb_trns
grad_emb_trns = grad_h_pre_bnorm @ W1.T
compare('emb_trns', grad_emb_trns, emb_trns)

# 25. gradient of loss w.r.t emb
grad_emb = grad_emb_trns.view(emb.shape)
compare('emb', grad_emb, emb)

# 26. gradient of loss w.r.t C
grad_C = torch.zeros_like(C)
for i in range(X_batch.shape[0]):
    for j in range(X_batch.shape[1]):
        ix = X_batch[i, j]
        grad_C[ix] += grad_emb[i, j]

compare('C', grad_C, C)