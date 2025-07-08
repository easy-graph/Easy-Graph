"""GitHub Users Social Network Dataset (musae_git)

This dataset represents a directed social network of GitHub users collected in 2019.
Nodes represent GitHub developers, and a directed edge from user A to user B indicates that A follows B.

Each node also includes:
- Features: User profile and activity-based features.
- Labels: Developer's project area (e.g., machine learning, web dev, etc.)

Statistics:
- Nodes: 37,700
- Edges: 289,003
- Feature dim: 5,575
- Classes: 2

Reference:
J. Leskovec et al. "SNAP Datasets: Stanford Large Network Dataset Collection",
https://snap.stanford.edu/data/github-social.html
"""

import csv
import json
import os

import easygraph as eg
import numpy as np

from easygraph.classes.graph import Graph

from .graph_dataset_base import EasyGraphBuiltinDataset
from .utils import download
from .utils import extract_archive


class GitHubUsersDataset(EasyGraphBuiltinDataset):
    r"""GitHub developers social graph (musae_git).

    Parameters
    ----------
    raw_dir : str, optional
        Directory to store raw data. Default: None
    force_reload : bool, optional
        Force re-download and processing. Default: False
    verbose : bool, optional
        Print processing information. Default: True
    transform : callable, optional
        Transform to apply to the graph on load.

    Examples
    --------
    >>> from easygraph.datasets import GitHubUsersDataset
    >>> dataset = GitHubUsersDataset()
    >>> g = dataset[0]
    >>> print("Nodes:", g.number_of_nodes())
    >>> print("Edges:", g.number_of_edges())
    >>> print("Feature shape:", g.nodes[0]['feat'].shape)
    >>> print("Label:", g.nodes[0]['label'])
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, transform=None):
        name = "musae_git"
        url = "https://snap.stanford.edu/data/git_web_ml.zip"
        super(GitHubUsersDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        archive = os.path.join(self.raw_dir, self.name + ".zip")
        download(self.url, path=archive)
        extract_archive(archive, self.raw_path)

    def process(self):
        g = eg.DiGraph()
        base_path = os.path.join(self.raw_path, "git_web_ml")

        # Load node features
        with open(os.path.join(base_path, "musae_git_features.json"), "r") as f:
            features = json.load(f)

        # Load labels
        labels = {}
        with open(os.path.join(base_path, "musae_git_target.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row["id"])
                labels[node_id] = int(row["ml_target"])

        # Load edges
        with open(os.path.join(base_path, "musae_git_edges.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u, v = int(row["id_1"]), int(row["id_2"])
                g.add_edge(u, v)

        # Add node attributes
        for node_id in g.nodes:
            feat = np.array(features[str(node_id)], dtype=np.float32)
            label = labels.get(node_id, -1)
            g.add_node(node_id, feat=feat, label=label)

        self._g = g
        self._num_classes = len(set(labels.values()))

        if self.verbose:
            print("Finished loading GitHub Users dataset.")
            print(f"  NumNodes: {g.number_of_nodes()}")
            print(f"  NumEdges: {g.number_of_edges()}")
            print(f"  Feature dim: {feat.shape[0]}")
            print(f"  NumClasses: {self._num_classes}")

    def __getitem__(self, idx):
        assert idx == 0, "GitHubUsersDataset only contains one graph"
        return self._g if self._transform is None else self._transform(self._g)

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes
