"""RoadNet-CA Dataset

This dataset represents the road network of California.
Nodes correspond to intersections, and edges represent roads connecting them.

The data is undirected and unweighted. No features or labels are provided.

Statistics:
- Nodes: 1,965,206
- Edges: 2,766,607
- Features: None
- Labels: None

Reference:
J. Leskovec and A. Krevl, “SNAP Datasets: Stanford Large Network Dataset Collection,”
https://snap.stanford.edu/data/roadNet-CA.html
"""

import gzip
import os
import shutil

import easygraph as eg

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import download


class RoadNetCADataset(EasyGraphBuiltinDataset):
    r"""Road network of California (RoadNet-CA)

    Nodes are road intersections and edges are roads connecting them.

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
    >>> from easygraph.datasets import RoadNetCADataset
    >>> dataset = RoadNetCADataset()
    >>> g = dataset[0]
    >>> print("Nodes:", g.number_of_nodes())
    >>> print("Edges:", g.number_of_edges())
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "roadNet-CA"
        url = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
        super(RoadNetCADataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        r"""Download and decompress the .txt.gz file."""
        compressed_path = os.path.join(self.raw_dir, self.name + ".txt.gz")
        extracted_path = os.path.join(self.raw_path, self.name + ".txt")

        download(self.url, path=compressed_path)

        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path)

        with gzip.open(compressed_path, "rb") as f_in:
            with open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    def process(self):
        graph = eg.Graph()  # Undirected road network
        edge_list_path = os.path.join(self.raw_path, self.name + ".txt")

        with open(edge_list_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                u, v = map(int, line.strip().split())
                graph.add_edge(u, v)

        self._g = graph
        self._num_nodes = graph.number_of_nodes()
        self._num_edges = graph.number_of_edges()

        if self.verbose:
            print("Finished loading RoadNet-CA dataset.")
            print(f"  NumNodes: {self._num_nodes}")
            print(f"  NumEdges: {self._num_edges}")

    def __getitem__(self, idx):
        assert idx == 0, "RoadNetCADataset only contains one graph"
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1
