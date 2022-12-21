import torch
import torch.nn as nn


class MultiHeadWrapper(nn.Module):
    r"""A wrapper to apply multiple heads to a given layer.

    Args:
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
