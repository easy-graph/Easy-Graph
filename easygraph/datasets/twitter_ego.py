import gzip
import os

import easygraph as eg

from easygraph.datasets.graph_dataset_base import EasyGraphBuiltinDataset
from easygraph.datasets.utils import download
from easygraph.datasets.utils import extract_archive


class TwitterEgoDataset(EasyGraphBuiltinDataset):
    r"""
    Twitter Ego Network Dataset

    The Twitter dataset was collected from public sources and contains a large ego-network of Twitter users.
    The combined network includes 81K edges among 81K users.

    Source: J. McAuley and J. Leskovec, Stanford SNAP, 2012
    URL: https://snap.stanford.edu/data/egonets-Twitter.html
    File used: https://snap.stanford.edu/data/twitter_combined.txt.gz
    """

    def __init__(self):
        super(TwitterEgoDataset, self).__init__(
            name="twitter_ego",
            url="https://snap.stanford.edu/data/twitter_combined.txt.gz",
            force_reload=False,
        )

    def download(self):
        gz_path = os.path.join(self.raw_path, "twitter_combined.txt.gz")
        download(self.url, path=gz_path)
        extract_archive(gz_path, self.raw_path)

    def process(self):
        import gzip

        import easygraph as eg

        gz_path = os.path.join(self.raw_path, "twitter_combined.txt.gz")
        txt_path = os.path.join(self.raw_path, "twitter_combined.txt")

        if not os.path.exists(txt_path):
            with gzip.open(gz_path, "rt") as f_in, open(txt_path, "w") as f_out:
                f_out.writelines(f_in)

        G = eg.Graph()
        edge_count = 0
        with open(txt_path, "r") as f:
            for line in f:
                u, v = map(int, line.strip().split())
                G.add_edge(u, v)
                edge_count += 1

        self._graphs = [G]
        self._graph = G
        self._processed = True

    def __getitem__(self, idx):
        if self._graph is not None:
            return self._graph
        elif self._graphs:
            return self._graphs[idx]
        else:
            return None
