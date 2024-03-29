import torch
import torch.nn as nn

from easygraph.classes import Hypergraph


class HGNNConv(nn.Module):
    r"""The HGNN convolution layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1}
        \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).

    where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{H}` is the hypergraph incidence matrix,
    :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, :math:`\mathbf{D}_v` is a diagonal vertex degree matrix,
    :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix, :math:`\mathbf{\Theta}` is the learnable parameters.

    Parameters:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

        # self.Theta1 = nn.Linear(in_size, hidden_dims)
        # self.Theta2 = nn.Linear(hidden_dims, out_size)
        # self.dropout = nn.Dropout(0.5)
        #
        # ###########################################################
        # # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        # ###########################################################
        # d_V = H.sum(1)  # node degree
        # d_E = H.sum(0)  # edge degree
        # n_edges = d_E.shape[0]
        # D_V_invsqrt = dglsp.diag(d_V ** -0.5)  # D_V ** (-1/2)
        # D_E_inv = dglsp.diag(d_E ** -1)  # D_E ** (-1)
        # W = dglsp.identity((n_edges, n_edges))
        # self.laplacian = D_V_invsqrt @ H @ W @ D_E_inv @ H.T @ D_V_invsqrt

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Parameters:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``eg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
