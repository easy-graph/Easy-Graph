from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser

import easygraph as eg
import numpy as np
<<<<<<< HEAD
import tensorflow as tf

from easygraph.utils import *


def get_relation_of_index_and_node(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes:
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def l_2nd(beta):
    try:
        pass
    except ImportWarning:
        print("tensorflow not found, please install")
    from tensorflow.python.keras import backend as K

    def loss_2nd(y_true, y_pred):
        y_true_numpy = y_true.numpy()
        b_ = np.ones_like(y_true.numpy())
        b_[y_true_numpy != 0] = beta
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.cast(y_pred, tf.int64)
        x = K.square((y_true - y_pred) * tf.cast(b_, tf.int64))
        t = K.sum(
            x,
            axis=-1,
        )
        return K.mean(t)

    return loss_2nd


def l_1st(alpha):
    try:
        import tensorflow as tf
    except ImportWarning:
        print("tensorflow not found, please install")
    from tensorflow.python.keras import backend as K

    def loss_1st(y_true, y_pred):
        L = y_true
        Y = y_pred
        batch_size = tf.cast(K.shape(L)[0], dtype=tf.float32)
        return (
            alpha
            * 2
            * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y))
            / batch_size
        )

    return loss_1st


def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    try:
        pass
    except ImportWarning:
        print("tensorflow not found, please install")
    from tensorflow.python.keras.layers import Dense
    from tensorflow.python.keras.layers import Input
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.regularizers import l1_l2

    A = Input(shape=(node_size,))
    L = Input(shape=(None,))
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(
                hidden_size[i],
                activation="relu",
                kernel_regularizer=l1_l2(l1, l2),
                name="1st",
            )(fc)
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torch.utils.data.dataloader import DataLoader


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )
    parser.add_argument(
        "--output", default="node.emb", help="Output representation file"
    )
    parser.add_argument(
        "--workers", default=8, type=int, help="Number of parallel processes."
    )
    parser.add_argument(
        "--weighted", action="store_true", default=False, help="Treat graph as weighted"
    )
    parser.add_argument(
        "--epochs", default=400, type=int, help="The training epochs of SDNE"
    )
    parser.add_argument(
        "--dropout",
        default=0.05,
        type=float,
        help="Dropout rate (1 - keep probability)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight for L2 loss on embedding matrix",
    )
    parser.add_argument("--lr", default=0.006, type=float, help="learning rate")
    parser.add_argument(
        "--alpha", default=1e-2, type=float, help="alhpa is a hyperparameter in SDNE"
    )
    parser.add_argument(
        "--beta", default=5.0, type=float, help="beta is a hyperparameter in SDNE"
    )
    parser.add_argument(
        "--nu1", default=1e-5, type=float, help="nu1 is a hyperparameter in SDNE"
    )
    parser.add_argument(
        "--nu2", default=1e-4, type=float, help="nu2 is a hyperparameter in SDNE"
    )
    parser.add_argument("--bs", default=100, type=int, help="batch size of SDNE")
    parser.add_argument("--nhid0", default=1000, type=int, help="The first dim")
    parser.add_argument("--nhid1", default=128, type=int, help="The second dim")
    parser.add_argument(
        "--step_size", default=10, type=int, help="The step size for lr"
    )
    parser.add_argument("--gamma", default=0.9, type=int, help="The gamma for lr")
    args = parser.parse_args()

    return args


class Dataload(data.Dataset):
    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node

    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat

    def __len__(self):
        return self.Node


def get_adj(g):
    edges = list(g.edges)
    edges = [(edges[i][0], edges[i][1]) for i in range(len(edges))]
    # print(edges)
    edges = np.array([np.array(i) for i in edges])
    min_node, max_node = edges.min(), edges.max()
    if min_node == 0:
        Node = max_node + 1
    else:
        Node = max_node

    Adj = np.zeros([Node, Node], dtype=int)
    for i in range(edges.shape[0]):
        g.add_edge(edges[i][0], edges[i][1])
        if min_node == 0:
            Adj[edges[i][0], edges[i][1]] = 1
            Adj[edges[i][1], edges[i][0]] = 1
>>>>>>> 622d76c2ce75db856dfd2eb6540dea6c9a7fe225
        else:
            Adj[edges[i][0] - 1, edges[i][1] - 1] = 1
            Adj[edges[i][1] - 1, edges[i][0] - 1] = 1
    Adj = torch.FloatTensor(Adj)
    return Adj, Node


class SDNE(nn.Module):
    """
    Graph embedding via SDNE.

        Parameters
        ----------
        graph : easygraph.Graph or easygraph.DiGraph

        node: Size of nodes

        nhid0, nhid1: Two dimensions of two hiddenlayers, default: 128, 64

        dropout: One parameter for regularization, default: 0.025

        alpha, beta:  Twe parameters
        graph=g: : easygraph.Graph or easygraph.DiGraph

    Examples
    --------
    >>> import easygraph as eg
    >>> model = eg.SDNE(graph=g, node_size= len(g.nodes), nhid0=128, nhid1=64, dropout=0.025, alpha=2e-2, beta=10)
    >>> emb = model.train(model, epochs, lr, bs, step_size, gamma, nu1, nu2, device, output)


    epochs,  "--epochs", default=400, type=int, help="The training epochs of SDNE"

    alpha,   "--alpha", default=2e-2, type=float, help="alhpa is a hyperparameter in SDNE"

    beta, "--beta", default=10.0, type=float, help="beta is a hyperparameter in SDNE"

    lr, "--lr", default=0.006, type=float, help="learning rate"

    bs, "--bs", default=100, type=int, help="batch size of SDNE"

    step_size,  "--step_size", default=10, type=int, help="The step size for lr"

    gamma, # "--gamma", default=0.9, type=int, help="The gamma for lr"

    step_size, "--step_size", default=10, type=int, help="The step size for lr"

    nu1, # "--nu1", default=1e-5, type=float, help="nu1 is a hyperparameter in SDNE"

    nu2,  "--nu2", default=1e-4, type=float, help="nu2 is a hyperparameter in SDNE"

    device, "-- device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") "

    output  "--output", default="node.emb", help="Output representation file"


    Reference
        ----------
        .. [1] Wang, D., Cui, P., & Zhu, W. (2016, August). Structural deep network embedding. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1225-1234).

        https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf
    """

    def __init__(
        self, graph, node_size, nhid0, nhid1, dropout=0.06, alpha=2e-2, beta=10.0
    ):
        super(SDNE, self).__init__()
        self.encode0 = nn.Linear(node_size, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size)
        self.droput = dropout
        self.alpha = alpha
        self.beta = beta
        self.graph = graph

    def forward(self, adj_batch, adj_mat, b_mat):
        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)
        L_1st = torch.sum(
            adj_mat
            * (
                embedding_norm
                - 2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                + torch.transpose(embedding_norm, dim0=0, dim1=1)
            )
        )
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))
        return L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd

    def train(
        self,
        model,
        epochs=100,
        lr=0.006,
        bs=100,
        step_size=10,
        gamma=0.9,
        nu1=1e-5,
        nu2=1e-4,
        device="cpu",
        output="out.emb",
    ):
        Adj, Node = get_adj(self.graph)
        model = model.to(device)

        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=step_size, gamma=gamma
        )
        Data = Dataload(Adj, Node)
        Data = DataLoader(
            Data,
            batch_size=bs,
            shuffle=True,
        )

        for epoch in range(1, epochs + 1):
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            for index in Data:
                adj_batch = Adj[index]
                adj_mat = adj_batch[:, index]
                b_mat = torch.ones_like(adj_batch)
                b_mat[adj_batch != 0] = self.beta

                opt.zero_grad()
                L_1st, L_2nd, L_all = model(adj_batch, adj_mat, b_mat)
                L_reg = 0
                for param in model.parameters():
                    L_reg += nu1 * torch.sum(torch.abs(param)) + nu2 * torch.sum(
                        param * param
                    )
                Loss = L_all + L_reg
                Loss.backward()
                opt.step()
                loss_sum += Loss
                loss_L1 += L_1st
                loss_L2 += L_2nd
                loss_reg += L_reg
            scheduler.step(epoch)
            # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
            print("loss for epoch %d is:" % epoch)
            print("loss_sum is %f" % loss_sum)
            print("loss_L1 is %f" % loss_L1)
            print("loss_L2 is %f" % loss_L2)
            print("loss_reg is %f" % loss_reg)

        # model.eval()
        embedding = model.savector(Adj)
        outVec = embedding.detach().numpy()
        np.savetxt(output, outVec)

        return outVec

    def savector(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0


# if __name__ == '__main__':
#     args = parse_args()
#     print(args)
#     dataset = eg.CiteseerGraphDataset(force_reload=True) # Download CiteseerGraphDataset contained in EasyGraph
#     num_classes = dataset.num_classes
#     g = dataset[0]
#     print(g)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     adj, node = get_adj(g)
#     # labels = g.ndata['label']
#     nhid0, nhid1, dropout, alpha = args.nhid0, args.nhid1, args.dropout, args.alpha
#     model = SDNE(node, nhid0, nhid1, dropout, alpha, graph=g)
#     print(model)
#
#     emb = model.train(args, device)
