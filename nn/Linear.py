import torch
class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self._weights = torch.randn((in_features, out_features)) / in_features ** 0.5  ## Random weights initialized using Kaiming's method
        self._bias    = torch.zeros(out_features) if bias else None

    def __call__(self, x):
        self.out = x @ self._weights
        if self._bias is not None:
            self.out += self._bias
        return self.out

    def parameters(self):
        return [self._weights] + ([] if self._bias is None else [self._bias])