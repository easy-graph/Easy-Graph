import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadWrapper(nn.Module):
    r"""A wrapper to apply multiple heads to a given layer.

    Parameters:
        ``num_heads`` (``int``): The number of heads.
        ``readout`` (``bool``): The readout method. Can be ``"mean"``, ``"max"``, ``"sum"``, or ``"concat"``.
        ``layer`` (``nn.Module``): The layer to apply multiple heads.
        ``**kwargs``: The keyword arguments for the layer.

    """

    def __init__(
        self, num_heads: int, readout: str, layer: nn.Module, **kwargs
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_heads):
            self.layers.append(layer(**kwargs))
        self.num_heads = num_heads
        self.readout = readout

    def forward(self, **kwargs) -> torch.Tensor:
        r"""The forward function.

        .. note::
            You must explicitly pass the keyword arguments to the layer. For example, if the layer is ``GATConv``, you must pass ``X=X`` and ``g=g``.
        """
        if self.readout == "concat":
            return torch.cat([layer(**kwargs) for layer in self.layers], dim=-1)
        else:
            outs = torch.stack([layer(**kwargs) for layer in self.layers])
            if self.readout == "mean":
                return outs.mean(dim=0)
            elif self.readout == "max":
                return outs.max(dim=0)[0]
            elif self.readout == "sum":
                return outs.sum(dim=0)
            else:
                raise ValueError("Unknown readout type")


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
        normalization="bn",
        InputNorm=False,
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert normalization in ["bn", "ln", "None"]
        if normalization == "bn":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif normalization == "ln":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is "Identity"):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
