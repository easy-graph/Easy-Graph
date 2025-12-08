import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# class GraphSAGEConv(nn.Module):
#     """
#     GraphSAGE convolution layer supporting 'mean' and 'pool' aggregation.

#     Parameters:
#         in_channels (int): Input feature dimension.
#         out_channels (int): Output feature dimension.
#         aggr (str): Aggregation method, either 'mean' or 'pool'.
#         bias (bool): Whether to add bias.
#         dropout (float): Dropout rate.
#         use_bn (bool): Whether to use BatchNorm1d.
#         is_last (bool): If True, skip activation and dropout.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         aggr: str = "mean",
#         bias: bool = True,
#         dropout: float = 0.5,
#         use_bn: bool = False,
#         is_last: bool = False,
#     ):
#         super(GraphSAGEConv, self).__init__()
#         assert aggr in ["mean", "pool"]
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.aggr = aggr
#         self.dropout = dropout
#         self.is_last = is_last
#         self.use_bn = use_bn

#         self.weight = Parameter(torch.Tensor(in_channels * 2, out_channels))
#         if aggr == "pool":
#             self.fc_pool = nn.Linear(in_channels, in_channels)

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         if self.use_bn:
#             self.bn = nn.BatchNorm1d(out_channels)
#         else:
#             self.bn = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_channels)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#         if self.aggr == "pool":
#             nn.init.xavier_uniform_(self.fc_pool.weight)

#     def forward(self, x, adj):
#         N = x.size(0)
#         if adj.is_sparse:
#             adj = adj.to_dense()

#         if self.aggr == "mean":
#             deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
#             agg = torch.matmul(adj, x) / deg  

#         elif self.aggr == "pool":
#             x_pool = F.relu(self.fc_pool(x)) 

#             masked = adj.unsqueeze(-1) * x_pool.unsqueeze(0) 
#             masked[adj == 0] = float('-inf')
#             agg = torch.max(masked, dim=1)[0]  

#         else:
#             raise NotImplementedError

#         h = torch.cat([x, agg], dim=1)         
#         h = torch.matmul(h, self.weight)     

#         if self.bias is not None:
#             h = h + self.bias

#         if not self.is_last:
#             h = F.relu(h)
#             if self.bn is not None:
#                 h = self.bn(h)
#             h = F.dropout(h, p=self.dropout, training=self.training)

#         return h

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.in_channels} -> {self.out_channels}, aggr='{self.aggr}')"

class GraphSAGEConv(nn.Module):
    """
    GraphSAGE convolution layer supporting 'mean' and 'pool' aggregation with SparseTensor.
    Uses g.adj_t for adjacency to maintain consistent API.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        bias: bool = True,
        dropout: float = 0.5,
        use_bn: bool = False,
        is_last: bool = False,
    ):
        super(GraphSAGEConv, self).__init__()
        assert aggr in ["mean", "pool"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.dropout = dropout
        self.is_last = is_last
        self.use_bn = use_bn

        self.weight = Parameter(torch.Tensor(in_channels * 2, out_channels))
        if aggr == "pool":
            self.fc_pool = nn.Linear(in_channels, in_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.aggr == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight)

    def forward(self, x, g):
        """
        x: (N, F_in)
        g: Graph object with g.adj_t as torch_sparse.SparseTensor
        """
        adj_t: SparseTensor = g.adj_t  # 从图对象取稀疏邻接矩阵

        if self.aggr == "mean":
            # 邻居聚合 (稀疏矩阵乘法)
            agg = adj_t.matmul(x)
            # 计算度并做平均
            deg = adj_t.sum(dim=1).clamp(min=1).unsqueeze(-1)
            agg = agg / deg

        elif self.aggr == "pool":
            # 非线性映射
            x_pool = F.relu(self.fc_pool(x))
            # 用 torch_scatter 实现基于邻居的 max pooling
            row, col, _ = adj_t.coo()  # 获取边索引
            agg = scatter(x_pool[col], row, dim=0, reduce='max')

        else:
            raise NotImplementedError

        # 拼接自身特征与聚合特征
        h = torch.cat([x, agg], dim=1)
        h = torch.matmul(h, self.weight)

        if self.bias is not None:
            h = h + self.bias

        if not self.is_last:
            h = F.relu(h)
            if self.bn is not None:
                h = self.bn(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h