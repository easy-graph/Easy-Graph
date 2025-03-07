import torch
import torch.nn.functional as F
import torch.nn as nn

from easygraph.nn import HWNNConv


class HWNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Parameters:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``ncount`` (``int``): The Number of node in the hypergraph.
        ``hyper_snapshot_num`` (``int``): The Number of snapshots splited from hypergraph.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
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
        self.convolution_1 = HWNNConv(in_channels, hid_channels, ncount, K1=3, K2=3, approx=True)
        self.convolution_2 = HWNNConv(hid_channels, num_classes, ncount, K1=3, K2=3, approx=True)
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
            Y = F.relu(self.convolution_1(X,hg))
            Y = F.dropout(Y, self.drop_rate)
            Y = self.convolution_2(Y,hg)
            Y = F.log_softmax(Y, dim=1)
            channel.append(Y)
        X = torch.zeros_like(channel[0])
        for ind in range(hyper_snapshot_num):
            X = X + self.par[ind] * channel[ind]
        return X