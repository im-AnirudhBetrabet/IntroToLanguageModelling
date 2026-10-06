import torch

class Sequential:
    def __init__(self, layers):
        self._layers = layers

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self._layers for p in layer.parameters()]