import torch
import torch.nn as nn
import torch.nn.functional as F

class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.5, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))

        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * heads if concat else out_channels))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, g):

        device = x.device
        N = x.size(0)
        H, C = self.heads, self.out_channels
        src, dst = g.edge_index

        h = torch.einsum('nf,hfc->nhc', x, self.weight)
        if self.training and self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=True)

        h_src = h[src]   
        h_dst = h[dst] 

        h_cat = torch.cat([h_src, h_dst], dim=-1)

        alpha = (h_cat * self.att.unsqueeze(0)).sum(dim=-1)
        alpha = self.leakyrelu(alpha)

        alpha_max = torch.full((N,H), -1e9, device=device)
        alpha_max.scatter_reduce_(0, dst[:,None].expand(-1,H), alpha, reduce="amax", include_self=True)
        alpha_exp = torch.exp(alpha - alpha_max[dst])
        sum_exp = torch.zeros(N,H,device=device).index_add_(0, dst, alpha_exp)
        alpha = alpha_exp / (sum_exp[dst] + 1e-16)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        out = torch.zeros(N,H,C,device=device)
        out.index_add_(0, dst, h_src * alpha.unsqueeze(-1))

        if self.concat:
            out = out.reshape(N, H*C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias.view(1,-1)

        return out
