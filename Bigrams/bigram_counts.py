"""
Character-level Bigram Language Model using Counts.

This script builds a character-level language model from the names dataset
using bigram frequencies.

It:
- Builds a vocabulary of characters with '.' as the start/end token.
- Counts how frequently each character is followed by another character.
- Converts the counts into probabilities with Laplace smoothing.
- Generates new names by sampling from the learned probabilities.
- Evaluates the model using log likelihood and negative log likelihood.

This implementation demonstrates the basic idea behind a bigram language
model before introducing neural networks.
"""

import torch
from pathlib import Path

N = torch.zeros([27, 27], dtype=torch.int32)                               # 27 x 27 matrix for the 26 letters and the start/end token
PARENT_DIR = Path(__file__).parent.parent

with open( PARENT_DIR / 'names.txt', 'r') as f:
    data = f.read().splitlines()


words       : list[str]              = sorted(list(set(''.join(data))))               # create a list of all available alphabets in the vocabulary sorted alphabetically
lookup_table: dict[str, int]         = {ch: idx + 1 for idx, ch in enumerate(words)}  # create a lookup dictionary for the alphabets using its 'index' in the words list
lookup_table['.']                    = 0                                              # create a lookup index for that 'start of word' character
reverse_lookup_table: dict[int, str] = {i:s for s,i in lookup_table.items()}          # Create a reverse lookup table where the index maps to the character

for w in data:
    chs = ['.'] + list(w) + ['.']                                         # prepend the start of word character and append the end of word character to the current word
    for w1, w2 in zip(chs, chs[1:]):                                      # create combinations of each character in the transformed word
        idx1 = lookup_table[w1]                                           # lookup index of w1 from the table
        idx2 = lookup_table[w2]                                           # lookup index of w2 from the table
        N[idx1, idx2] += 1


P = (N + 1).float()                                                       # Cast counts to floats and add 1 for Laplace smoothing to avoid zero probabilities
P /= P.sum(dim=1, keepdim=True)                                           # Create a probability distribution


gen = torch.Generator().manual_seed(2147483647)
for i in range(10):
    idx  = 0
    word = []
    while True:
        p   = P[idx]
        idx = torch.multinomial(p, num_samples=1, generator=gen, replacement=True).item()
        gen_word = reverse_lookup_table[idx]
        word.append(gen_word)
        if idx == 0:
            break

    print("".join(word))

log_likelihood = 0.0
n = 0
for w in data:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1      = lookup_table[ch1]
        ix2      = lookup_table[ch2]
        prob     = P[ix1, ix2]
        log_prob = torch.log(prob)
        n       += 1
        log_likelihood += log_prob                          # log(a*b*c) = log(a) + log(b) + log(c)

negative_log_likelihood = log_likelihood * -1
print(f'Log Likelihood            = {log_likelihood:.4f}')
print(f'Negative Log Likelihood   = {negative_log_likelihood:.4f}')
print(f'Average Negative Log Likelihood = {(negative_log_likelihood / n):.4f}')

