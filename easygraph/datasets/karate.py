import easygraph as eg

from .graph_dataset_base import EasyGraphDataset
from .utils import _set_labels
from .utils import tensor


""" KarateClubDataset for inductive learning. """


class KarateClubDataset(EasyGraphDataset):
    """Karate Club dataset for Node Classification

    Zachary's karate club is a social network of a university
    karate club, described in the paper "An Information Flow
    Model for Conflict and Fission in Small Groups" by Wayne W. Zachary.
    The network became a popular example of community structure in
    networks after its use by Michelle Girvan and Mark Newman in 2002.
    Official website: `<http://konect.cc/networks/ucidata-zachary/>`_

    Karate Club dataset statistics:

    - Nodes: 34
    - Edges: 156
    - Number of Classes: 2

    Parameters
    ----------
    transform : callable, optional
        A transform that takes in a :class:`~eg.Graph` object and returns
        a transformed version. The :class:`~eg.Graph` object will be
        transformed before every access.

    Attributes
    ----------
    num_classes : int
        Number of node classes

    Examples
    --------
    >>> dataset = KarateClubDataset()
    >>> num_classes = dataset.num_classes
    >>> g = dataset[0]
    >>> labels = g.ndata['label']
    """

    def __init__(self, transform=None):
        super(KarateClubDataset, self).__init__(name="karate_club", transform=transform)

    def process(self):
        import numpy as np

        kc_graph = eg.get_graph_karateclub()
        label = np.asarray(
            [kc_graph.nodes[i]["club"] != "Mr. Hi" for i in kc_graph.nodes]
        ).astype(np.int64)
        label = tensor(label)

        kc_graph = _set_labels(kc_graph, label)
        kc_graph.ndata["label"] = label
        self._graph = kc_graph
        self._data = [kc_graph]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    def __getitem__(self, idx):
        r"""Get graph object

        Parameters
        ----------
        idx : int
            Item index, KarateClubDataset has only one graph object

        Returns
        -------
        :class:`eg.Graph`

            graph structure and labels.

            - ``ndata['label']``: ground truth labels
        """
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1
