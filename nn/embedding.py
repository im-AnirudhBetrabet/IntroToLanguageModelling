import torch

class Embedding:
    def __init__(self, num_embeddings, embedding_dimensions):
        self._weights = torch.randn((num_embeddings, embedding_dimensions))

    def __call__(self, ix):
        self.out = self._weights[ix]
        return self.out

    def parameters(self):
        return [self._weights]