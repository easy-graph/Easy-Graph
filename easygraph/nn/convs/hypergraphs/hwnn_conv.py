import torch
import torch.nn as nn

from easygraph.classes import Hypergraph


class HWNNConv(nn.Module):
    r"""The HWNNConv model proposed in `Heterogeneous Hypergraph Embedding for Graph Classification <https://arxiv.org/pdf/2010.10728>`_ paper (WSDM 2021).

    Parameters:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``ncount`` (``int``): The Number of node in the hypergraph.
        ``K1`` (``int``): Polynomial calculation times.
        ``K2`` (``int``): Polynomial calculation times.
        ``approx`` (``bool``): Whether to use polynomial fitting
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ncount: int,
        K1: int = 2,
        K2: int = 2,
        approx: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K1 = K1
        self.K2 = K2
        self.ncount = ncount
        self.approx = approx
        self.W = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.W_d = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.uniform_(self.W_d, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Parameters:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``eg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        if self.approx == True:
            X = hg.smoothing_with_HWNN_approx(
                X, self.par, self.W_d, self.K1, self.K2, self.W
            )
        else:
            X = hg.smoothing_with_HWNN_wavelet(X, self.W_d, self.W)
        return X
