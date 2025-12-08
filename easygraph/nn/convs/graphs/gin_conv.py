import torch
import torch.nn as nn

class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0.0, train_eps=True):
        super().__init__()
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, g):

        agg = g.adj_t.matmul(x)

        out = (1 + self.eps) * x + agg

        out = self.mlp(out)
        return out
