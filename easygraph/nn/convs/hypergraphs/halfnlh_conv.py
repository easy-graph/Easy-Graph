import torch
import torch.nn as nn
import torch.nn.functional as F

from easygraph.nn.convs.common import MLP
from easygraph.nn.convs.pma import PMA
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


class HalfNLHconv(MessagePassing):
    r"""The HalfNLHconv model proposed in `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper (ICLR 2022).

    Parameters:
        ``in_dim`` (``int``): : The dimension of input.
        ``hid_dim`` (``int``): : The dimension of hidden.
        ``out_dim`` (``int``): : The dimension of output.
        ``num_layers`` (``int``): : The number of layers.
        ``dropout`` (``float``): Dropout ratio. Defaults to 0.5.
        ``normalization`` (``str``): The normalization method. Defaults to ``bn``
        ``InputNorm`` (``bool``): Defaults to False.
        ``heads`` (``int``):  Defaults to 1
        `attention`` (``bool``):  Defaults to True

    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout,
        normalization="bn",
        InputNorm=False,
        heads=1,
        attention=True,
    ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(
                    in_dim,
                    hid_dim,
                    hid_dim,
                    num_layers,
                    dropout,
                    normalization,
                    InputNorm,
                )
                self.f_dec = MLP(
                    hid_dim,
                    hid_dim,
                    out_dim,
                    num_layers,
                    dropout,
                    normalization,
                    InputNorm,
                )
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    def reset_parameters(self):
        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ is "Identity"):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ is "Identity"):
                self.f_dec.reset_parameters()

    #         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr="add"):
        """
        input -> MLP -> Prop
        """

        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None, aggr="sum"):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()

        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)
