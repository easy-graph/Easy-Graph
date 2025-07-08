"""Web-Google Dataset

This dataset is a web graph based on Google's web pages and their hyperlink
structure, as crawled by the Stanford WebBase project in 2002.

Each node represents a web page, and a directed edge from u to v indicates
a hyperlink from page u to page v.

Statistics:
- Nodes: 875713
- Edges: 5105039
- Features: None
- Labels: None

Reference:
J. Leskovec, A. Rajaraman, J. Ullman, “Mining of Massive Datasets.”
Dataset from SNAP: https://snap.stanford.edu/data/web-Google.html
"""

import gzip
import os
import shutil

import easygraph as eg

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import download
from .utils import extract_archive


class WebGoogleDataset(EasyGraphBuiltinDataset):
    r"""Web-Google hyperlink network dataset.

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
    >>> from easygraph.datasets import WebGoogleDataset
    >>> dataset = WebGoogleDataset()
    >>> g = dataset[0]
    >>> print("Nodes:", g.number_of_nodes())
    >>> print("Edges:", g.number_of_edges())
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "web-Google"
        url = "https://snap.stanford.edu/data/web-Google.txt.gz"
        super(WebGoogleDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        r"""Download and extract .gz edge list."""
        if self.url is not None:
            file_path = os.path.join(self.raw_dir, self.name + ".txt.gz")
            download(self.url, path=file_path)
            extract_archive(file_path, self.raw_path)

    def process(self):
        graph = eg.DiGraph()  # Web-Google is directed
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
            print("Finished loading Web-Google dataset.")
            print(f"  NumNodes: {self._num_nodes}")
            print(f"  NumEdges: {self._num_edges}")

    def __getitem__(self, idx):
        assert idx == 0, "WebGoogleDataset only contains one graph"
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1

    def download(self):
        r"""Download and decompress the .txt.gz file."""
        if self.url is not None:
            compressed_path = os.path.join(self.raw_dir, self.name + ".txt.gz")
            extracted_path = os.path.join(self.raw_path, self.name + ".txt")

            # Download .gz file
            download(self.url, path=compressed_path)

            # Ensure output directory exists
            if not os.path.exists(self.raw_path):
                os.makedirs(self.raw_path)

            # Decompress manually
            with gzip.open(compressed_path, "rb") as f_in:
                with open(extracted_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
