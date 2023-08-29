import torch
import torch.nn as nn


class DHNE(nn.Module):
    r"""The DHNE model proposed in `Structural Deep Embedding for Hyper-Networks <https://arxiv.org/abs/1711.10146>`_ paper (AAAI 2018).

    Parameters:
        ``dim_feature`` (``int``): : feature dimension list ( len = 3)
        ``embedding_size`` (``int``): :The embedding dimension size
        ``hidden_size`` (``int``): The hidden full connected layer size.

    """

    def __init__(self, dim_feature, embedding_size, hidden_size):
        super(DHNE, self).__init__()
        self.dim_feature = dim_feature
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.encode0 = nn.Sequential(
            nn.Linear(
                in_features=self.dim_feature[0], out_features=self.embedding_size[0]
            )
        )
        self.encode1 = nn.Sequential(
            nn.Linear(
                in_features=self.dim_feature[1], out_features=self.embedding_size[1]
            )
        )
        self.encode2 = nn.Sequential(
            nn.Linear(
                in_features=self.dim_feature[2], out_features=self.embedding_size[2]
            )
        )
        self.decode_layer0 = nn.Linear(
            in_features=self.embedding_size[0], out_features=self.dim_feature[0]
        )
        self.decode_layer1 = nn.Linear(
            in_features=self.embedding_size[1], out_features=self.dim_feature[1]
        )
        self.decode_layer2 = nn.Linear(
            in_features=self.embedding_size[2], out_features=self.dim_feature[2]
        )

        self.hidden_layer = nn.Linear(
            in_features=sum(self.embedding_size), out_features=self.hidden_size
        )
        self.ouput_layer = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, input0, input1, input2):
        input0 = self.encode0(input0)
        input0 = torch.tanh(input0)
        decode0 = self.decode_layer0(input0)
        decode0 = torch.sigmoid(decode0)

        input1 = self.encode1(input1)
        input1 = torch.tanh(input1)
        decode1 = self.decode_layer1(input1)
        decode1 = torch.sigmoid(decode1)

        input2 = self.encode2(input2)
        input2 = torch.tanh(input2)
        decode2 = self.decode_layer2(input2)
        decode2 = torch.sigmoid(decode2)

        merged = torch.tanh(torch.cat((input0, input1, input2), dim=1))
        merged = self.hidden_layer(merged)
        merged = self.ouput_layer(merged)
        merged = torch.sigmoid(merged)
        return [decode0, decode1, decode2, merged]
