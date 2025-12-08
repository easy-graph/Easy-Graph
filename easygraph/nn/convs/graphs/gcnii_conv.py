import torch
import torch.nn as nn

class GCNIIConv(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.1, theta=0.5, bias=True, learnable_alpha=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta = theta

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
        else:
            self.register_buffer('alpha', torch.Tensor([alpha]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if hasattr(self, 'bias') and self.bias is not None:
            nn.init.zeros_(self.bias)
        if hasattr(self, 'alpha') and isinstance(self.alpha, nn.Parameter):
            nn.init.constant_(self.alpha, self.alpha.item())

    def forward(self, x, g, x0=None):

        if x0 is None:
            x0 = x

        agg = g.adj_t.matmul(x @ self.weight) 

        out = (1 - self.alpha) * agg + self.alpha * x0

        out = self.theta * out

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, alpha={}, theta={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.alpha.item(), self.theta
        )
