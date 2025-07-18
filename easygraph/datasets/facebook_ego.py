"""Facebook Ego-Net Dataset

This dataset contains a subset of Facebook’s social network collected from
survey participants in the SNAP EgoNet project. Nodes represent users, and
edges indicate friendship links between them.

Each ego network is centered on a user and includes their friend connections
and friend-to-friend connections. The `.circles` files contain labeled groups
(i.e., communities) of friends identified by the ego user.

This version processes all ego-nets as a single undirected graph. Node features
are not provided. Labels (circles) are optional and not included by default.

Statistics (based on merged graph):
- Nodes: ~4,000+
- Edges: ~88,000+
- Features: None
- Classes: None

Reference:
J. McAuley and J. Leskovec, “Learning to Discover Social Circles in Ego Networks,”
in NIPS, 2012. [https://snap.stanford.edu/data/egonets-Facebook.html]
"""

import os

import easygraph as eg

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import download
from .utils import extract_archive


class FacebookEgoNetDataset(EasyGraphBuiltinDataset):
    r"""Facebook Ego-Net social network dataset.

    Each node is a user, and edges represent friendship. The dataset
    includes 10 ego networks centered on different users.

    Parameters
    ----------
    raw_dir : str, optional
        Directory to store the raw downloaded files. Default: None
    force_reload : bool, optional
        Whether to re-download and process the dataset. Default: False
    verbose : bool, optional
        Whether to print detailed processing logs. Default: True
    transform : callable, optional
        Optional transform to apply on the graph.

    Examples
    --------
    >>> from easygraph.datasets import FacebookEgoNetDataset
    >>> dataset = FacebookEgoNetDataset()
    >>> g = dataset[0]
    >>> print("Nodes:", g.number_of_nodes())
    >>> print("Edges:", g.number_of_edges())
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "facebook"
        url = "https://snap.stanford.edu/data/facebook.tar.gz"
        super(FacebookEgoNetDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        parent_dir = os.path.join(self.raw_path, "facebook")
        g = eg.Graph()

        # Iterate over all .edges files in the subdirectory
        for filename in os.listdir(parent_dir):
            if filename.endswith(".edges"):
                edge_file = os.path.join(parent_dir, filename)

                with open(edge_file, "r") as f:
                    for line in f:
                        u, v = map(int, line.strip().split())
                        g.add_edge(u, v)

        self._g = g
        self._num_nodes = g.number_of_nodes()
        self._num_edges = g.number_of_edges()

        if self.verbose:
            print("Finished loading Facebook Ego-Net dataset.")
            print(f"  NumNodes: {self._num_nodes}")
            print(f"  NumEdges: {self._num_edges}")

    def __getitem__(self, idx):
        assert idx == 0, "FacebookEgoNetDataset only contains one merged graph"
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1

    def download(self):
        r"""Automatically download data and extract it."""
        if self.url is not None:
            archive_path = os.path.join(self.raw_dir, self.name + ".tar.gz")
            download(self.url, path=archive_path)
            extract_archive(archive_path, self.raw_path)
