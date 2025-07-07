import time
import warnings

import easygraph as eg
import numpy as np
import torch
import torch.nn as nn

from easygraph.utils import alias_draw
from easygraph.utils import alias_setup
from sklearn import preprocessing

# from easygraph.functions.graph_embedding import *
from tqdm import tqdm


warnings.filterwarnings("ignore")


class LINE(nn.Module):
    """Graph embedding via LINE.
    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph
    dimension: int
    walk_length: int

    walk_num: int

    negative: int
    batch_size: int

    init_alpha: float
    order: int
    Returns
    -------
    embedding_vector : dict
        The embedding vector of each node
    Examples
    --------
    >>> model = LINE(
    ...          dimension=128,
    ...          walk_length=80,
    ...          walk_num=20,
    ...          negative=5,
    ...          batch_size=128,
    ...          init_alpha=0.025,
    ...          order=3  )
    >>> model.train()
    >>> emb = model(g, return_dict=True) # g: easygraph.Graph or easygraph.DiGraph

    References
    ----------

    .. [1] Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015, May). Line: Large-scale information network embedding. In Proceedings of the 24th international conference on world wide web (pp. 1067-1077).

    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp0228-Tang.pdf

    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--walk-length",
            type=int,
            default=80,
            help="Length of walk per source. Default is 80.",
        )
        parser.add_argument(
            "--walk-num",
            type=int,
            default=20,
            help="Number of walks per source. Default is 20.",
        )
        parser.add_argument(
            "--negative",
            type=int,
            default=5,
            help="Number of negative node in sampling. Default is 5.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="Batch size in SGD training process. Default is 1000.",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.025,
            help="Initial learning rate of SGD. Default is 0.025.",
        )
        parser.add_argument(
            "--order",
            type=int,
            default=3,
            help="Order of proximity in LINE. Default is 3 for 1+2.",
        )
        parser.add_argument("--hidden-size", type=int, default=128)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.walk_length,
            args.walk_num,
            args.negative,
            args.batch_size,
            args.alpha,
            args.order,
        )

    def __init__(
        self,
        dimension=128,
        walk_length=80,
        walk_num=20,
        negative=5,
        batch_size=128,
        init_alpha=0.025,
        order=3,
    ):
        super(LINE, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.negative = negative
        self.batch_size = batch_size
        self.init_alpha = init_alpha
        self.order = order

    def forward(self, g, return_dict=True):
        # run LINE algorithm, 1-order, 2-order or 3(1-order + 2-order)

        self.G = g
        self.is_directed = g.is_directed()
        self.num_node = len(g.nodes)
        self.num_edge = g.number_of_edges()
        self.num_sampling_edge = self.walk_length * self.walk_num * self.num_node

        node2id = dict([(node, vid) for vid, node in enumerate(g.nodes)])
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges]

        self.edges_prob = np.asarray([1.0 for e in g.edges])
        self.edges_prob /= np.sum(self.edges_prob)
        self.edges_table, self.edges_prob = alias_setup(self.edges_prob)

        degree_weight = np.asarray([0] * self.num_node)
        degree_weight = np.array(list(g.degree(node2id[u] for u in g.nodes).values()))
        # for u,v in g.edges:

        #     degree_weight[node2id[u]] += 1.0
        #     if not self.is_directed:
        #         degree_weight[node2id[v]] += 1.0
        self.node_prob = np.power(degree_weight, 0.75)
        self.node_prob /= np.sum(self.node_prob)
        self.node_table, self.node_prob = alias_setup(self.node_prob)

        if self.order == 3:
            self.dimension = int(self.dimension / 2)
        if self.order == 1 or self.order == 3:
            print("train line with 1-order")
            print(type(self.dimension))
            self.emb_vertex = (
                np.random.random((self.num_node, self.dimension)) - 0.5
            ) / self.dimension
            self._train_line(order=1)
            embedding1 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 2 or self.order == 3:
            print("train line with 2-order")
            self.emb_vertex = (
                np.random.random((self.num_node, self.dimension)) - 0.5
            ) / self.dimension
            self.emb_context = self.emb_vertex
            self._train_line(order=2)
            embedding2 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 1:
            embeddings = embedding1
        elif self.order == 2:
            embeddings = embedding2
        else:
            print("concatenate two embedding...")
            embeddings = np.hstack((embedding1, embedding2))

        if return_dict:
            features_matrix = dict()
            for vid, node in enumerate(g.nodes):
                features_matrix[node] = embeddings[vid]
        else:
            features_matrix = np.zeros((len(g.nodes), embeddings.shape[1]))
            nx_nodes = list(g.nodes)
            features_matrix[nx_nodes] = embeddings[np.arange(len(g.nodes))]
        return features_matrix

    def _update(self, vec_u, vec_v, vec_error, label):
        # update vetex embedding and vec_error
        f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
        g = (self.alpha * (label - f)).reshape((len(label), 1))
        vec_error += g * vec_v
        vec_v += g * vec_u

    def _train_line(self, order):
        # train Line model with order
        self.alpha = self.init_alpha
        batch_size = self.batch_size
        t0 = time.time()
        num_batch = int(self.num_sampling_edge / batch_size)
        epoch_iter = tqdm(range(num_batch))
        for b in epoch_iter:
            if b % 100 == 0:
                epoch_iter.set_description(
                    #    f"Progress: {b * 1.0 / num_batch * 100:.4f}, alpha: {self.alpha:.6f}, time: {time.time() - t0:.4f}"
                )
                self.alpha = self.init_alpha * max((1 - b * 1.0 / num_batch), 0.0001)
            u, v = [0] * batch_size, [0] * batch_size
            for i in range(batch_size):
                edge_id = alias_draw(self.edges_table, self.edges_prob)
                u[i], v[i] = self.edges[edge_id]
                if not self.is_directed and np.random.rand() > 0.5:
                    v[i], u[i] = self.edges[edge_id]

            vec_error = np.zeros((batch_size, self.dimension))
            label, target = np.asarray([1 for i in range(batch_size)]), np.asarray(v)
            for j in range(1 + self.negative):
                if j != 0:
                    label = np.asarray([0 for i in range(batch_size)])
                    for i in range(batch_size):
                        target[i] = alias_draw(self.node_table, self.node_prob)
                if order == 1:
                    self._update(
                        self.emb_vertex[u], self.emb_vertex[target], vec_error, label
                    )
                else:
                    self._update(
                        self.emb_vertex[u], self.emb_context[target], vec_error, label
                    )
            self.emb_vertex[u] += vec_error


if __name__ == "__main__":
    dataset = eg.CiteseerGraphDataset(
        force_reload=True
    )  # Download CiteseerGraphDataset contained in EasyGraph
    num_classes = dataset.num_classes
    g = dataset[0]
    labels = g.ndata["label"]
    edge_list = []
    for i in g.edges:
        edge_list.append((i[0], i[1]))
    g1 = eg.Graph()
    g1.add_edges_from(edge_list)
    # print(g.edges)
    # print(g.__dir__())

    model = LINE(
        dimension=128,
        walk_length=80,
        walk_num=20,
        negative=5,
        batch_size=128,
        init_alpha=0.025,
        order=3,
    )
    print(model)

    model.train()
    out = model(g1, return_dict=True)

    keylist = sorted(out)
    tmp = torch.cat(
        (
            torch.unsqueeze(torch.tensor(out[keylist[0]]), -2),
            torch.unsqueeze(torch.tensor(out[keylist[1]]), -2),
        ),
        0,
    )

    for i in range(2, len(keylist)):
        tmp = torch.cat((tmp, torch.unsqueeze(torch.tensor(out[keylist[i]]), -2)), 0)
    torch.save(tmp, "line.emb")
    print(tmp, tmp.shape)

    line_emb = []
    for i in range(0, len(tmp)):
        line_emb.append(list(tmp[i]))
    line_emb = np.array(line_emb)

# tsne = TSNE(n_components=2)
# z = tsne.fit_transform(line_emb)
# z_data = np.vstack((z.T, labels)).T
# df_tsne = pd.DataFrame(z_data, columns=['Dim1', 'Dim2', 'class'])
# df_tsne['class'] = df_tsne['class'].astype(int)
# df_tsne.head()
#
# plt.figure(figsize=(8, 8))
# sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2', palette=['green','orange','brown','red', 'blue','black'])
# plt.savefig('torch_line_citeseer.pdf', bbox_inches='tight')
# plt.show()
#
#
