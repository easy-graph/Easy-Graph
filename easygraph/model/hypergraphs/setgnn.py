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

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance * norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
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


# import os
# import time
# import torch
# import argparse
#
# import numpy as np
# import os.path as osp

# import matplotlib.pyplot as plt
#
# from tqdm import tqdm
#
#
# from easygraph.model.data.convert_datasets_to_pygDataset import dataset_Hypergraph
# from torch_scatter import scatter_add
#
# def parse_method(args, data):
#     #     Currently we don't set hyperparameters w.r.t. different dataset
#     if args.method == 'AllSetTransformer':
#         if args.LearnMask:
#             model = SetGNN(args, data.norm)
#         else:
#             model = SetGNN(num_classes=args.num_classes,num_features=args.num_features)
#
#     elif args.method == 'AllDeepSets':
#         args.PMA = False
#         args.aggregate = 'add'
#         if args.LearnMask:
#             model = SetGNN(args, data.norm)
#         else:
#             model = SetGNN(args)
#
#     #     elif args.method == 'SetGPRGNN':
#     #         model = SetGPRGNN(args)
#
#
#     return model
#
#
# class Logger(object):
#     """ Adapted from https://github.com/snap-stanford/ogb/ """
#
#     def __init__(self, runs, info=None):
#         self.info = info
#         self.results = [[] for _ in range(runs)]
#
#     def add_result(self, run, result):
#         assert len(result) == 3
#         assert run >= 0 and run < len(self.results)
#         self.results[run].append(result)
#
#     def print_statistics(self, run=None):
#         if run is not None:
#             result = 100 * torch.tensor(self.results[run])
#             argmax = result[:, 1].argmax().item()
#             print(f'Run {run + 1:02d}:')
#             print(f'Highest Train: {result[:, 0].max():.2f}')
#             print(f'Highest Valid: {result[:, 1].max():.2f}')
#             print(f'  Final Train: {result[argmax, 0]:.2f}')
#             print(f'   Final Test: {result[argmax, 2]:.2f}')
#         else:
#             result = 100 * torch.tensor(self.results)
#
#             best_results = []
#             for r in result:
#                 train1 = r[:, 0].max().item()
#                 valid = r[:, 1].max().item()
#                 train2 = r[r[:, 1].argmax(), 0].item()
#                 test = r[r[:, 1].argmax(), 2].item()
#                 best_results.append((train1, valid, train2, test))
#
#             best_result = torch.tensor(best_results)
#
#             print(f'All runs:')
#             r = best_result[:, 0]
#             print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
#             r = best_result[:, 1]
#             print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
#             r = best_result[:, 2]
#             print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
#             r = best_result[:, 3]
#             print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
#
#             return best_result[:, 1], best_result[:, 3]
#
#     def plot_result(self, run=None):
#         plt.style.use('seaborn')
#         if run is not None:
#             result = 100 * torch.tensor(self.results[run])
#             x = torch.arange(result.shape[0])
#             plt.figure()
#             print(f'Run {run + 1:02d}:')
#             plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
#             plt.legend(['Train', 'Valid', 'Test'])
#         else:
#             result = 100 * torch.tensor(self.results[0])
#             x = torch.arange(result.shape[0])
#             plt.figure()
#             #             print(f'Run {run + 1:02d}:')
#             plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
#             plt.legend(['Train', 'Valid', 'Test'])
#
#
# @torch.no_grad()
# def evaluate(model, data, split_idx, eval_func, result=None):
#     if result is not None:
#         out = result
#     else:
#         model.eval()
#         out = model(data)
#         out = F.log_softmax(out, dim=1)
#
#     train_acc = eval_func(
#         data.y[split_idx['train']], out[split_idx['train']])
#     valid_acc = eval_func(
#         data.y[split_idx['valid']], out[split_idx['valid']])
#     test_acc = eval_func(
#         data.y[split_idx['test']], out[split_idx['test']])
#
#     #     Also keep track of losses
#     train_loss = F.nll_loss(
#         out[split_idx['train']], data.y[split_idx['train']])
#     valid_loss = F.nll_loss(
#         out[split_idx['valid']], data.y[split_idx['valid']])
#     test_loss = F.nll_loss(
#         out[split_idx['test']], data.y[split_idx['test']])
#     return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out
#
#
# def eval_acc(y_true, y_pred):
#     acc_list = []
#     y_true = y_true.detach().cpu().numpy()
#     y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
#
#     #     ipdb.set_trace()
#     #     for i in range(y_true.shape[1]):
#     is_labeled = y_true == y_true
#     correct = y_true[is_labeled] == y_pred[is_labeled]
#     acc_list.append(float(np.sum(correct)) / len(correct))
#
#     return sum(acc_list) / len(acc_list)
#
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# # --- Main part of the training ---
# # # Part 0: Parse arguments
#
#
# """
#
# """
# def ExtractV2E(data):
#     # Assume edge_index = [V|E;E|V]
#     edge_index = data.edge_index
# #     First, ensure the sorting is correct (increasing along edge_index[0])
#     _, sorted_idx = torch.sort(edge_index[0])
#     edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
#     print("data.n_x:",data.n_x)
#     # num_nodes = data.n_x[0]
#     num_nodes = data.n_x
#     num_hyperedges = data.num_hyperedges
#     if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
#         print('num_hyperedges does not match! 1')
#         return
#     cidx = torch.where(edge_index[0] == num_nodes)[
#         0].min()  # cidx: [V...|cidx E...]
#     data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
#     return data
#
# def Add_Self_Loops(data):
#     # update so we dont jump on some indices
#     # Assume edge_index = [V;E]. If not, use ExtractV2E()
#     edge_index = data.edge_index
#     num_nodes = data.n_x
#     num_hyperedges = data.num_hyperedges
#
#     if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
#         print('num_hyperedges does not match! 2')
#         return
#
#     hyperedge_appear_fre = Counter(edge_index[1].numpy())
#     # store the nodes that already have self-loops
#     skip_node_lst = []
#     for edge in hyperedge_appear_fre:
#         if hyperedge_appear_fre[edge] == 1:
#             skip_node = edge_index[0][torch.where(
#                 edge_index[1] == edge)[0].item()]
#             skip_node_lst.append(skip_node.item())
#
#     new_edge_idx = edge_index[1].max() + 1
#     new_edges = torch.zeros(
#         (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
#     tmp_count = 0
#     for i in range(num_nodes):
#         if i not in skip_node_lst:
#             new_edges[0][tmp_count] = i
#             new_edges[1][tmp_count] = new_edge_idx
#             new_edge_idx += 1
#             tmp_count += 1
#
#     data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
#     edge_index = torch.cat((edge_index, new_edges), dim=1)
#     # Sort along w.r.t. nodes
#     _, sorted_idx = torch.sort(edge_index[0])
#     data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
#     return data
#
# def expand_edge_index(data, edge_th=0):
#     '''
#     args:
#         num_nodes: regular nodes. i.e. x.shape[0]
#         num_edges: number of hyperedges. not the star expansion edges.
#
#     this function will expand each n2he relations, [[n_1, n_2, n_3],
#                                                     [e_7, e_7, e_7]]
#     to :
#         [[n_1,   n_1,   n_2,   n_2,   n_3,   n_3],
#          [e_7_2, e_7_3, e_7_1, e_7_3, e_7_1, e_7_2]]
#
#     and each he2n relations:   [[e_7, e_7, e_7],
#                                 [n_1, n_2, n_3]]
#     to :
#         [[e_7_1, e_7_2, e_7_3],
#          [n_1,   n_2,   n_3]]
#
#     and repeated for every hyperedge.
#     '''
#     edge_index = data.edge_index
#     num_nodes = data.n_x[0].item()
#     if hasattr(data, 'totedges'):
#         num_edges = data.totedges
#     else:
#         num_edges = data.num_hyperedges[0]
#
#     expanded_n2he_index = []
# #     n2he_with_same_heid = []
#
# #     expanded_he2n_index = []
# #     he2n_with_same_heid = []
#
#     # start edge_id from the largest node_id + 1.
#     cur_he_id = num_nodes
#     # keep an mapping of new_edge_id to original edge_id for edge_size query.
#     new_edge_id_2_original_edge_id = {}
#
#     # do the expansion for all annotated he_id in the original edge_index
# #     ipdb.set_trace()
#     for he_idx in range(num_nodes, num_edges + num_nodes):
#         # find all nodes within the same hyperedge.
#         selected_he = edge_index[:, edge_index[1] == he_idx]
#         size_of_he = selected_he.shape[1]
#
# #         Trim a hyperedge if its size>edge_th
#         if edge_th > 0:
#             if size_of_he > edge_th:
#                 continue
#
#         if size_of_he == 1:
#             # there is only one node in this hyperedge -> self-loop node. add to graph.
#             #             n2he_with_same_heid.append(selected_he)
#
#             new_n2he = selected_he.clone()
#             new_n2he[1] = cur_he_id
#             expanded_n2he_index.append(new_n2he)
#
#             # ====
# #             new_he2n_same_heid = torch.flip(selected_he, dims = [0])
# #             he2n_with_same_heid.append(new_he2n_same_heid)
#
# #             new_he2n = torch.flip(selected_he, dims = [0])
# #             new_he2n[0] = cur_he_id
# #             expanded_he2n_index.append(new_he2n)
#
#             cur_he_id += 1
#             continue
#
#         # -------------------------------
# #         # new_n2he_same_heid uses same he id for all nodes.
# #         new_n2he_same_heid = selected_he.repeat_interleave(size_of_he - 1, dim = 1)
# #         n2he_with_same_heid.append(new_n2he_same_heid)
#
#         # for new_n2he mapping. connect the nodes to all repeated he first.
#         # then remove those connection that corresponding to the node itself.
#         new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)
#
#         # new_edge_ids start from the he_id from previous iteration (cur_he_id).
#         new_edge_ids = torch.LongTensor(
#             np.arange(cur_he_id, cur_he_id + size_of_he)).repeat(size_of_he)
#         new_n2he[1] = new_edge_ids
#
#         # build a mapping between node and it's corresponding edge.
#         # e.g. {n_1: e_7_1, n_2: e_7_2}
#         tmp_node_id_2_he_id_dict = {}
#         for idx in range(size_of_he):
#             new_edge_id_2_original_edge_id[cur_he_id] = he_idx
#             cur_node_id = selected_he[0][idx].item()
#             tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
#             cur_he_id += 1
#
#         # create n2he by deleting the self-product edge.
#         new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
#         for col_idx in range(new_n2he.shape[1]):
#             tmp_node_id, tmp_edge_id = new_n2he[0, col_idx].item(
#             ), new_n2he[1, col_idx].item()
#             if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
#                 new_he_select_mask[col_idx] = False
#         new_n2he = new_n2he[:, new_he_select_mask]
#         expanded_n2he_index.append(new_n2he)
#
#
# #         # ---------------------------
# #         # create he2n from mapping.
# #         new_he2n = np.array([[he_id, node_id] for node_id, he_id in tmp_node_id_2_he_id_dict.items()])
# #         new_he2n = torch.from_numpy(new_he2n.T).to(device = edge_index.device)
# #         expanded_he2n_index.append(new_he2n)
#
# #         # create he2n with same heid as input edge_index.
# #         new_he2n_same_heid = torch.zeros_like(new_he2n, device = edge_index.device)
# #         new_he2n_same_heid[1] = new_he2n[1]
# #         new_he2n_same_heid[0] = torch.ones_like(new_he2n[0]) * he_idx
# #         he2n_with_same_heid.append(new_he2n_same_heid)
#
#     new_edge_index = torch.cat(expanded_n2he_index, dim=1)
# #     new_he2n_index = torch.cat(expanded_he2n_index, dim = 1)
# #     new_edge_index = torch.cat([new_n2he_index, new_he2n_index], dim = 1)
#     # sort the new_edge_index by first row. (node_ids)
#     new_order = new_edge_index[0].argsort()
#     data.edge_index = new_edge_index[:, new_order]
#
#     return data
#
# def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
#     """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
#     """ randomly splits label into train/valid/test splits """
#     if not balance:
#         if ignore_negative:
#             labeled_nodes = torch.where(label != -1)[0]
#         else:
#             labeled_nodes = label
#
#         n = labeled_nodes.shape[0]
#         train_num = int(n * train_prop)
#         valid_num = int(n * valid_prop)
#
#         perm = torch.as_tensor(np.random.permutation(n))
#
#         train_indices = perm[:train_num]
#         val_indices = perm[train_num:train_num + valid_num]
#         test_indices = perm[train_num + valid_num:]
#
#         if not ignore_negative:
#             return train_indices, val_indices, test_indices
#
#         train_idx = labeled_nodes[train_indices]
#         valid_idx = labeled_nodes[val_indices]
#         test_idx = labeled_nodes[test_indices]
#
#         split_idx = {'train': train_idx,
#                      'valid': valid_idx,
#                      'test': test_idx}
#     else:
#         #         ipdb.set_trace()
#         indices = []
#         for i in range(label.max()+1):
#             index = torch.where((label == i))[0].view(-1)
#             index = index[torch.randperm(index.size(0))]
#             indices.append(index)
#
#         percls_trn = int(train_prop/(label.max()+1)*len(label))
#         val_lb = int(valid_prop*len(label))
#         train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
#         rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
#         rest_index = rest_index[torch.randperm(rest_index.size(0))]
#         valid_idx = rest_index[:val_lb]
#         test_idx = rest_index[val_lb:]
#         split_idx = {'train': train_idx,
#                      'valid': valid_idx,
#                      'test': test_idx}
#     return split_idx
#
# def norm_contruction(data, option='all_one', TYPE='V2E'):
#     from torch_geometric.nn.conv.gcn_conv import gcn_norm
#     if TYPE == 'V2E':
#         if option == 'all_one':
#             data.norm = torch.ones_like(data.edge_index[0])
#
#         elif option == 'deg_half_sym':
#             edge_weight = torch.ones_like(data.edge_index[0])
#             cidx = data.edge_index[1].min()
#             Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
#             HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
#             V_norm = Vdeg**(-1/2)
#             E_norm = HEdeg**(-1/2)
#             data.norm = V_norm[data.edge_index[0]] * \
#                 E_norm[data.edge_index[1]-cidx]
#
#     elif TYPE == 'V2V':
#         data.edge_index, data.norm = gcn_norm(
#             data.edge_index, data.norm, add_self_loops=True)
#     return data
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_prop', type=float, default=0.5)
#     parser.add_argument('--valid_prop', type=float, default=0.25)
#     parser.add_argument('--dname', default='cora')
#     # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
#     parser.add_argument('--method', default='AllSetTransformer')
#     parser.add_argument('--epochs', default=500, type=int)
#     # Number of runs for each split (test fix, only shuffle train/val)
#     parser.add_argument('--runs', default=20, type=int)
#     parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
#     parser.add_argument('--dropout', default=0.5, type=float)
#     parser.add_argument('--lr', default=0.001, type=float)
#     parser.add_argument('--wd', default=0.0, type=float)
#     # How many layers of full NLConvs
#     parser.add_argument('--All_num_layers', default=2, type=int)
#     parser.add_argument('--MLP_num_layers', default=2,
#                         type=int)  # How many layers of encoder
#     parser.add_argument('--MLP_hidden', default=64,
#                         type=int)  # Encoder hidden units
#     parser.add_argument('--Classifier_num_layers', default=2,
#                         type=int)  # How many layers of decoder
#     parser.add_argument('--Classifier_hidden', default=64,
#                         type=int)  # Decoder hidden units
#     parser.add_argument('--display_step', type=int, default=-1)
#     parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
#     # ['all_one','deg_half_sym']
#     parser.add_argument('--normtype', default='all_one')
#     parser.add_argument('--add_self_loop', action='store_false')
#     # NormLayer for MLP. ['bn','ln','None']
#     parser.add_argument('--normalization', default='ln')
#     parser.add_argument('--deepset_input_norm', default=True)
#     parser.add_argument('--GPR', action='store_false')  # skip all but last dec
#     # skip all but last dec
#     parser.add_argument('--LearnMask', action='store_false')
#     parser.add_argument('--num_features', default=0, type=int)  # Placeholder
#     parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
#     # Choose std for synthetic feature noise
#     parser.add_argument('--feature_noise', default='1', type=str)
#     # whether the he contain self node or not
#     parser.add_argument('--exclude_self', action='store_true')
#     parser.add_argument('--PMA', action='store_true')
#     #     Args for HyperGCN
#     parser.add_argument('--HyperGCN_mediators', action='store_true')
#     parser.add_argument('--HyperGCN_fast', action='store_true')
#     #     Args for Attentions: GAT and SetGNN
#     parser.add_argument('--heads', default=1, type=int)  # Placeholder
#     parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
#     #     Args for HNHN
#     parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
#     parser.add_argument('--HNHN_beta', default=-0.5, type=float)
#     parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
#     #     Args for HCHA
#     parser.add_argument('--HCHA_symdegnorm', action='store_true')
#     #     Args for UniGNN
#     parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
#     parser.add_argument('--UniGNN_degV', default=0)
#     parser.add_argument('--UniGNN_degE', default=0)
#
#     parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
#     parser.set_defaults(add_self_loop=True)
#     parser.set_defaults(exclude_self=False)
#     parser.set_defaults(GPR=False)
#     parser.set_defaults(LearnMask=False)
#     parser.set_defaults(HyperGCN_mediators=True)
#     parser.set_defaults(HyperGCN_fast=True)
#     parser.set_defaults(HCHA_symdegnorm=False)
#
#     #     Use the line below for .py file
#     args = parser.parse_args()
#     #     Use the line below for notebook
#     # args = parser.parse_args([])
#     # args, _ = parser.parse_known_args()
#
#     # # Part 1: Load data
#
#     ### Load and preprocess data ###
#     existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
#                         'NTU2012', 'Mushroom',
#                         'coauthor_cora', 'coauthor_dblp',
#                         'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
#                         'walmart-trips-100', 'house-committees-100',
#                         'cora', 'citeseer', 'pubmed']
#
#     synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100',
#                       'house-committees-100']
#
#     if args.dname in existing_dataset:
#         dname = args.dname
#         f_noise = args.feature_noise
#         if (f_noise is not None) and dname in synthetic_list:
#             p2raw = '../data/raw_data/AllSet_all_raw_data/'
#             dataset = dataset_Hypergraph(name=dname,
#                                          feature_noise=f_noise,
#                                          p2raw=p2raw)
#         else:
#             if dname in ['cora', 'citeseer', 'pubmed']:
#                 p2raw = '../data/cocitation/'
#             elif dname in ['coauthor_cora', 'coauthor_dblp']:
#                 p2raw = '../data/AllSet_all_raw_data/coauthorship/'
#             elif dname in ['yelp']:
#                 p2raw = '../data/AllSet_all_raw_data/yelp/'
#             else:
#                 p2raw = '../data/AllSet_all_raw_data/'
#             dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset_updated/',
#                                          p2raw=p2raw)
#         data = dataset.data
#         args.num_features = dataset.num_features
#         args.num_classes = dataset.num_classes
#         if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
#             #         Shift the y label to start with 0
#             args.num_classes = len(data.y.unique())
#             data.y = data.y - data.y.min()
#         if not hasattr(data, 'n_x'):
#             data.n_x = torch.tensor([data.x.shape[0]])
#         if not hasattr(data, 'num_hyperedges'):
#             # note that we assume the he_id is consecutive.
#             data.num_hyperedges = torch.tensor(
#                 [data.edge_index[0].max() - data.n_x[0] + 1])
#
#     # ipdb.set_trace()
#     #     Preprocessing
#     # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
#     if args.method in ['AllSetTransformer', 'AllDeepSets']:
#         data = ExtractV2E(data)
#         if args.add_self_loop:
#             data = Add_Self_Loops(data)
#         if args.exclude_self:
#             data = expand_edge_index(data)
#
#         data = norm_contruction(data, option=args.normtype)
#
#
#         #     Get splits
#     split_idx_lst = []
#     for run in range(args.runs):
#         split_idx = rand_train_test_idx(
#             data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
#         split_idx_lst.append(split_idx)
#
#     # # Part 2: Load model
#
#     model = parse_method(args, data)
#     # put things to device
#     if args.cuda in [0, 1]:
#         device = torch.device('cuda:' + str(args.cuda)
#                               if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device('cpu')
#
#     model, data = model.to(device), data.to(device)
#     if args.method == 'UniGCNII':
#         args.UniGNN_degV = args.UniGNN_degV.to(device)
#         args.UniGNN_degE = args.UniGNN_degE.to(device)
#
#     num_params = count_parameters(model)
#
#     # # Part 3: Main. Training + Evaluation
#
#     logger = Logger(args.runs, args)
#
#     criterion = nn.NLLLoss()
#     eval_func = eval_acc
#
#     model.train()
#     # print('MODEL:', model)
#
#     ### Training loop ###
#     runtime_list = []
#     for run in tqdm(range(args.runs)):
#         start_time = time.time()
#         split_idx = split_idx_lst[run]
#         train_idx = split_idx['train'].to(device)
#         model.reset_parameters()
#         if args.method == 'UniGCNII':
#             optimizer = torch.optim.Adam([
#                 dict(params=model.reg_params, weight_decay=0.01),
#                 dict(params=model.non_reg_params, weight_decay=5e-4)
#             ], lr=0.01)
#         else:
#             optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#         #     This is for HNHN only
#         #     if args.method == 'HNHN':
#         #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
#         best_val = float('-inf')
#         for epoch in range(args.epochs):
#             #         Training part
#             model.train()
#             optimizer.zero_grad()
#             out = model(data)
#             out = F.log_softmax(out, dim=1)
#             loss = criterion(out[train_idx], data.y[train_idx])
#             loss.backward()
#             optimizer.step()
#             #         if args.method == 'HNHN':
#             #             scheduler.step()
#             #         Evaluation part
#             result = evaluate(model, data, split_idx, eval_func)
#             logger.add_result(run, result[:3])
#
#             if epoch % args.display_step == 0 and args.display_step > 0:
#                 print(f'Epoch: {epoch:02d}, '
#                       f'Train Loss: {loss:.4f}, '
#                       f'Valid Loss: {result[4]:.4f}, '
#                       f'Test  Loss: {result[5]:.4f}, '
#                       f'Train Acc: {100 * result[0]:.2f}%, '
#                       f'Valid Acc: {100 * result[1]:.2f}%, '
#                       f'Test  Acc: {100 * result[2]:.2f}%')
#
#         end_time = time.time()
#         runtime_list.append(end_time - start_time)
#
#         # logger.print_statistics(run)
#
#     ### Save results ###
#     avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
#
#     best_val, best_test = logger.print_statistics()
#     res_root = 'hyperparameter_tunning'
#     if not osp.isdir(res_root):
#         os.makedirs(res_root)
#
#     filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
#     print(f"Saving results to {filename}")
#     with open(filename, 'a+') as write_obj:
#         cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}'
#         cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
#         cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
#         cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s'
#         cur_line += f',{avg_time // 60}min{(avg_time % 60):.2f}s'
#         cur_line += f'\n'
#         write_obj.write(cur_line)
#
#     all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
#     with open(all_args_file, 'a+') as f:
#         f.write(str(args))
#         f.write('\n')
#
#     print('All done! Exit python code')
#     quit()
