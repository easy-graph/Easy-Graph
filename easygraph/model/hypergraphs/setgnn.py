from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from easygraph.nn.convs.common import MLP
from easygraph.nn.convs.hypergraphs.halfnlh_conv import HalfNLHconv
from torch.nn import Linear


__all__ = ["SetGNN"]


class SetGNN(nn.Module):
    r"""The SetGNN model proposed in `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper (ICLR 2022).

    Parameters:
        ``num_features`` (``int``): : The dimension of node features.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``Classifier_hidden`` (``int``): Decoder hidden units.
        ``Classifier_num_layers`` (``int``): Layers of decoder.
        ``MLP_hidden`` (``int``): Encoder hidden units.
        ``MLP_num_layers`` (``int``): Layers of encoder.
         ``dropout`` (``float``, optional): Dropout ratio. Defaults to 0.5.
        ``aggregate`` (``str``): The aggregation method. Defaults to ``add``
        ``normalization`` (``str``): The normalization method. Defaults to ``ln``
        ``deepset_input_norm`` (``bool``):  Defaults to True.
        ``heads`` (``int``):  Defaults to 1
        `PMA`` (``bool``):  Defaults to True
        `GPR`` (``bool``):  Defaults to False
        `LearnMask`` (``bool``):  Defaults to False
        `norm`` (``Tensor``):  The weight for edges in bipartite graphs, correspond to data.edge_index

    """

    def __init__(
        self,
        num_features,
        num_classes,
        Classifier_hidden=64,
        Classifier_num_layers=2,
        MLP_hidden=64,
        MLP_num_layers=2,
        All_num_layers=2,
        dropout=0.5,
        aggregate="mean",
        normalization="ln",
        deepset_input_norm=True,
        heads=1,
        PMA=True,
        GPR=False,
        LearnMask=False,
        norm=None,
        self_loop=True,
    ):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

        #         Now set all dropout the same, but can be different
        self.All_num_layers = All_num_layers
        self.dropout = dropout
        self.aggr = aggregate
        self.NormLayer = normalization
        self.InputNorm = deepset_input_norm
        self.GPR = GPR
        self.LearnMask = LearnMask
        #         Now define V2EConvs[i], V2EConvs[i] for ith layers
        #         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
        #         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()
        self.edge_index = None
        self.self_loop = self_loop
        if self.LearnMask:
            self.Importance = nn.Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(
                in_channels=num_features,
                hidden_channels=Classifier_hidden,
                out_channels=num_classes,
                num_layers=Classifier_num_layers,
                dropout=self.dropout,
                normalization=self.NormLayer,
                InputNorm=False,
            )
        else:
            self.V2EConvs.append(
                HalfNLHconv(
                    in_dim=num_features,
                    hid_dim=MLP_hidden,
                    out_dim=MLP_hidden,
                    num_layers=MLP_num_layers,
                    dropout=self.dropout,
                    normalization=self.NormLayer,
                    InputNorm=self.InputNorm,
                    heads=heads,
                    attention=PMA,
                )
            )
            self.bnV2Es.append(nn.BatchNorm1d(MLP_hidden))
            self.E2VConvs.append(
                HalfNLHconv(
                    in_dim=MLP_hidden,
                    hid_dim=MLP_hidden,
                    out_dim=MLP_hidden,
                    num_layers=MLP_num_layers,
                    dropout=self.dropout,
                    normalization=self.NormLayer,
                    InputNorm=self.InputNorm,
                    heads=heads,
                    attention=PMA,
                )
            )
            self.bnE2Vs.append(nn.BatchNorm1d(MLP_hidden))
            for _ in range(self.All_num_layers - 1):
                self.V2EConvs.append(
                    HalfNLHconv(
                        in_dim=MLP_hidden,
                        hid_dim=MLP_hidden,
                        out_dim=MLP_hidden,
                        num_layers=MLP_num_layers,
                        dropout=self.dropout,
                        normalization=self.NormLayer,
                        InputNorm=self.InputNorm,
                        heads=heads,
                        attention=PMA,
                    )
                )
                self.bnV2Es.append(nn.BatchNorm1d(MLP_hidden))
                self.E2VConvs.append(
                    HalfNLHconv(
                        in_dim=MLP_hidden,
                        hid_dim=MLP_hidden,
                        out_dim=MLP_hidden,
                        num_layers=MLP_num_layers,
                        dropout=self.dropout,
                        normalization=self.NormLayer,
                        InputNorm=self.InputNorm,
                        heads=heads,
                        attention=PMA,
                    )
                )
                self.bnE2Vs.append(nn.BatchNorm1d(MLP_hidden))

            if self.GPR:
                self.MLP = MLP(
                    in_channels=num_features,
                    hidden_channels=MLP_hidden,
                    out_channels=MLP_hidden,
                    num_layers=MLP_num_layers,
                    dropout=self.dropout,
                    normalization=self.NormLayer,
                    InputNorm=False,
                )
                self.GPRweights = Linear(self.All_num_layers + 1, 1, bias=False)
                self.classifier = MLP(
                    in_channels=MLP_hidden,
                    hidden_channels=Classifier_hidden,
                    out_channels=num_classes,
                    num_layers=Classifier_num_layers,
                    dropout=self.dropout,
                    normalization=self.NormLayer,
                    InputNorm=False,
                )
            else:
                self.classifier = MLP(
                    in_channels=MLP_hidden,
                    hidden_channels=Classifier_hidden,
                    out_channels=num_classes,
                    num_layers=Classifier_num_layers,
                    dropout=self.dropout,
                    normalization=self.NormLayer,
                    InputNorm=False,
                )

    def generate_edge_index(self, dataset, self_loop=False):
        edge_list = dataset["edge_list"]
        e_ind = 0
        edge_index = [[], []]
        for e in edge_list:
            for n in e:
                edge_index[0].append(n)
                edge_index[1].append(e_ind)
            e_ind += 1
        edge_index = torch.tensor(edge_index).type(torch.LongTensor)
        if self_loop:
            hyperedge_appear_fre = Counter(edge_index[1].numpy())
            skip_node_lst = []
            for edge in hyperedge_appear_fre:
                if hyperedge_appear_fre[edge] == 1:
                    skip_node = edge_index[0][torch.where(edge_index[1] == edge)[0]]
                    skip_node_lst.append(skip_node)
            num_nodes = dataset["num_vertices"]
            new_edge_idx = len(edge_index[1]) + 1
            new_edges = torch.zeros(
                (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype
            )
            tmp_count = 0
            for i in range(num_nodes):
                if i not in skip_node_lst:
                    new_edges[0][tmp_count] = i
                    new_edges[1][tmp_count] = new_edge_idx
                    new_edge_idx += 1
                    tmp_count += 1

            edge_index = torch.Tensor(edge_index).type(torch.LongTensor)
            edge_index = torch.cat((edge_index, new_edges), dim=1)
            _, sorted_idx = torch.sort(edge_index[0])
            edge_index = torch.Tensor(edge_index[:, sorted_idx]).type(torch.LongTensor)

        return edge_index

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        if self.edge_index is None:
            self.edge_index = self.generate_edge_index(data, self.self_loop)
        # print("generate_edge_index:", self.edge_index.shape)
        x, edge_index = data["features"], self.edge_index
        if data["weight"] == None:
            norm = torch.ones(edge_index.size()[1])
        else:
            norm = data["weight"]

        if self.LearnMask:
            norm = self.Importance * norm

        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                #                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr))
                #                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)

        return x
