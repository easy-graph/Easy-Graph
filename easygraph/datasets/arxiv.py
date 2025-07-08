"""Arxiv HEP-TH Citation Network

This dataset represents the citation network of preprints from the High Energy Physics - Theory (HEP-TH) category on arXiv, covering the period from January 1993 to April 2003.

Each node corresponds to a paper, and a directed edge from paper A to paper B indicates that A cites B.

No features or labels are included in this dataset.

Statistics:
- Nodes: 27,770
- Edges: 352,807
- Features: None
- Labels: None

Reference:
J. Leskovec, J. Kleinberg and C. Faloutsos, "Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations,"
in KDD 2005. Dataset: https://snap.stanford.edu/data/cit-HepTh.html
"""

import gzip
import os
import shutil

import easygraph as eg

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import download


class ArxivHEPTHDataset(EasyGraphBuiltinDataset):
    r"""Arxiv HEP-TH citation network dataset.

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
    >>> from easygraph.datasets import ArxivHEPTHDataset
    >>> dataset = ArxivHEPTHDataset()
    >>> g = dataset[0]
    >>> print("Nodes:", g.number_of_nodes())
    >>> print("Edges:", g.number_of_edges())
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "cit-HepTh"
        url = "https://snap.stanford.edu/data/cit-HepTh.txt.gz"
        super(ArxivHEPTHDataset, self).__init__(
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
        graph = eg.DiGraph()  # Citation network is directed
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
            print("Finished loading Arxiv HEP-TH dataset.")
            print(f"  NumNodes: {self._num_nodes}")
            print(f"  NumEdges: {self._num_edges}")

    def __getitem__(self, idx):
        assert idx == 0, "ArxivHEPTHDataset only contains one graph"
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1
