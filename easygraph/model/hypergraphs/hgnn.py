import torch
import torch.nn as nn

from easygraph.nn import HGNNConv


class HGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Parameters:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Parameters:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``eg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        # import time
        # start = time.time()
        # X = self.laplacian @ self.Theta1(self.dropout(X))
        # end = time.time()
        # # print("lal:",end-start)
        # X = self.act(X)
        # X = self.laplacian @ self.Theta2(X)
        # return X

        for layer in self.layers:
            X = layer(X, hg)
        return X
