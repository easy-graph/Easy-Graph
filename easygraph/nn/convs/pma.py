import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easygraph.nn.convs.common import MLP
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class PMA(MessagePassing):
    """
    PMA part:
    Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
    i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
    In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels,
        hid_dim,
        out_channels,
        num_layers,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=False,
        **kwargs,
    ):
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.aggr = "add"
        #         self.input_seed = input_seed

        #         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
        #         Now, no seed as input. Directly learn the importance weights alpha_ij.
        #         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads * self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(
            in_channels=self.heads * self.hidden,
            hidden_channels=self.heads * self.hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=0.0,
            normalization="None",
        )
        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)
        #         if bias and concat:
        #             self.bias = Parameter(torch.Tensor(heads * out_channels))
        #         elif bias and not concat:
        #             self.bias = Parameter(torch.Tensor(out_channels))
        #         else:

        #         Always no bias! (For now)
        self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        #         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    #         zeros(self.bias)

    def forward(
        self, x, edge_index: Adj, size: Size = None, return_attention_weights=None
    ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None

        #         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
        #         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(self, x_j, alpha_j, index, ptr, size_j):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max() + 1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggr="add"):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggr is None:
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
