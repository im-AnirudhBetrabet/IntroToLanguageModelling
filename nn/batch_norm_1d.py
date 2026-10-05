import torch

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self._eps      = eps                  # Small value added to the variance for numerical stability
        self._momentum = momentum             # determines how much of the current mean and standard deviation will be added to the running values
        self.training  = True
        self._gamma    = torch.ones(dim)      # Scaling parameter
        self._beta     = torch.zeros(dim)     # Shifting parameter

        self._running_mean = torch.zeros(dim) # Stores the running mean
        self._running_var  = torch.ones(dim)  # Stores the running variance

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar  = x.var(0, keepdim=True)
        else:
            xmean = self._running_mean
            xvar  = self._running_var

        x_norm = (x - xmean) / torch.sqrt(xvar + self._eps)
        self.out = self._gamma * x_norm + self._beta

        if self.training:
            with torch.no_grad():
                self._running_mean = (((1 - self._momentum) * self._running_mean) + (self._momentum * xmean))
                self._running_var  = (((1 - self._momentum) * self._running_var)  + (self._momentum * xvar))

        return self.out

    def parameters(self):
        return [self._gamma, self._beta]