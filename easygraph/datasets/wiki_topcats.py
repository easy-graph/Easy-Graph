"""Wikipedia Top Categories Dataset (wiki-topcats)

This dataset is a directed graph of Wikipedia articles restricted to
top-level categories (at least 100 articles), capturing the largest
strongly connected component.

Statistics:
- Nodes: 1,791,489
- Edges: 28,511,807
- Categories: 17,364
- Overlapping labels per node

Source:
H. Yin, A. Benson, J. Leskovec, D. Gleich.
"Local Higher-order Graph Clustering", KDD 2017
Data: https://snap.stanford.edu/data/wiki-topcats.html
"""

import gzip
import os

import easygraph as eg

from easygraph.datasets.graph_dataset_base import EasyGraphBuiltinDataset
from easygraph.datasets.utils import download
from easygraph.datasets.utils import extract_archive


class WikiTopCatsDataset(EasyGraphBuiltinDataset):
    """Wikipedia Top Categories Snapshot from 2011 (SNAP)"""

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        super(WikiTopCatsDataset, self).__init__(
            name="wiki_topcats",
            url="https://snap.stanford.edu/data/wiki-topcats.txt.gz",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        # Download the main graph file
        gz_path = os.path.join(self.raw_dir, "wiki-topcats.txt.gz")
        download(self.url, path=gz_path)

        # Also download category info and page names
        cat_url = "https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz"
        names_url = "https://snap.stanford.edu/data/wiki-topcats-page-names.txt.gz"
        download(
            cat_url, path=os.path.join(self.raw_dir, "wiki-topcats-categories.txt.gz")
        )
        download(
            names_url, path=os.path.join(self.raw_dir, "wiki-topcats-page-names.txt.gz")
        )

    def process(self):
        raw = self.raw_dir

        # Decompress and read edges
        edge_gz = os.path.join(raw, "wiki-topcats.txt.gz")
        edge_txt = os.path.join(raw, "wiki-topcats.txt")
        if not os.path.exists(edge_txt):
            with gzip.open(edge_gz, "rt") as fin, open(edge_txt, "w") as fout:
                fout.writelines(fin)
        G = eg.DiGraph()
        edge_count = 0
        with open(edge_txt, "r") as f:
            for line in f:
                u, v = map(int, line.strip().split())
                G.add_edge(u, v)
                edge_count += 1
        if self.verbose:
            print(f"Loaded graph: {G.number_of_nodes()} nodes, {edge_count} edges")

        # Compress node names
        names_gz = os.path.join(raw, "wiki-topcats-page-names.txt.gz")
        names = {}
        with gzip.open(names_gz, "rt") as f:
            for idx, line in enumerate(f):
                names[idx] = line.strip()

        # Load categories
        cats_gz = os.path.join(raw, "wiki-topcats-categories.txt.gz")
        labels = {}  # mapping: node -> list of category strings
        with gzip.open(cats_gz, "rt") as f:
            for idx, line in enumerate(f):
                categories = line.strip().split(";")
                categories = [cat.strip() for cat in categories if cat.strip()]
                labels[idx] = categories

        # Attach node features: empty, and node labels
        for n in G.nodes:
            G.add_node(n, name=names.get(n, ""), label=labels.get(n, []))

        self._graph = G
        self._graphs = [G]
        self._processed = True

    def __getitem__(self, idx):
        assert idx == 0
        return self._graph

    def __len__(self):
        return 1
