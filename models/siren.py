import torch
import torch.nn as nn

class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, layers, w0):
        super().__init__()
        net = [nn.Linear(in_dim, hidden), Sine(w0)]
        for _ in range(layers):
            net += [nn.Linear(hidden, hidden), Sine(w0)]
        net.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)