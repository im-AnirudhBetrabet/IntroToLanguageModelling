import torch
def build_dataset(words, block_size, lookup_table):

    x, y       = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            idx = lookup_table[ch]
            x.append(context)
            y.append(idx)
            context = context[1:] + [idx]
    X = torch.tensor(x)
    Y = torch.tensor(y)
    print(f"Shape of X is {X.shape} , Shape of Y is {Y.shape}")
    return X, Y
