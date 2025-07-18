import torch
import torch.nn as nn
import torch.nn.functional as F

from easygraph.nn import HWNNConv


class HWNN(nn.Module):
    r"""The HWNN model proposed in `Heterogeneous Hypergraph Embedding for Graph Classification <https://arxiv.org/abs/2010.10728>`_ paper (WSDM 2021).

    Parameters:
        ``in_channels`` (``int``): Number of input feature channels. :math:`C_{in}` is the dimension of input features.
        ``num_classes`` (``int``): Number of target classes for classification.
        ``ncount`` (``int``): Total number of nodes in the hypergraph.
        ``hyper_snapshot_num`` (``int``, optional): number of sementic snapshots for the given heterogeneous hypergraph.
        ``hid_channels`` (``int``, optional): Number of hidden units. :math:`C_{hid}` is the dimension of hidden representations. Defaults to 128.
        ``drop_rate`` (``float``, optional): Dropout probability for regularization. Defaults to 0.01.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        ncount: int,
        hyper_snapshot_num: int = 1,
        hid_channels: int = 128,
        drop_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.convolution_1 = HWNNConv(
            in_channels, hid_channels, ncount, K1=3, K2=3, approx=True
        )
        self.convolution_2 = HWNNConv(
            hid_channels, num_classes, ncount, K1=3, K2=3, approx=True
        )
        self.par = torch.nn.Parameter(torch.Tensor(hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, X: torch.Tensor, hgs: list) -> torch.Tensor:
        r"""The forward function.

        Parameters:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``eg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
            ``hgs`` (``list`` of ``Hypergraph``): A list of hypergraph structures whcih stands for snapshots.
        """
        channel = []
        hyper_snapshot_num = len(hgs)
        for snap_index in range(hyper_snapshot_num):
            hg = hgs[snap_index]
            Y = F.relu(self.convolution_1(X, hg))
            Y = F.dropout(Y, self.drop_rate)
            Y = self.convolution_2(Y, hg)
            Y = F.log_softmax(Y, dim=1)
            channel.append(Y)
        X = torch.zeros_like(channel[0])
        for ind in range(hyper_snapshot_num):
            X = X + self.par[ind] * channel[ind]
        return X
