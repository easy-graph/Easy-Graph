from typing import Optional

from easygraph.datapipe import load_from_pickle
from easygraph.datapipe import to_long_tensor
from easygraph.datapipe import to_tensor
from easygraph.datasets.hypergraph.hypergraph_dataset_base import BaseData


class YelpRestaurant(BaseData):
    r"""The Yelp-Restaurant dataset is a restaurant-review network dataset for node classification task.

    More details see the DHG or `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper.

    The content of the Yelp-Restaurant dataset includes the following:

    - ``num_classes``: The number of classes: :math:`11`.
    - ``num_vertices``: The number of vertices: :math:`50,758`.
    - ``num_edges``: The number of edges: :math:`679,302`.
    - ``dim_features``: The dimension of features: :math:`1,862`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(50,758 \times 1,862)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`679,302`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(50,758, )`.
    - ``state``: The state list. ``torch.LongTensor`` with size :math:`(50,758, )`.
    - ``city``: The city list. ``torch.LongTensor`` with size :math:`(50,758, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("yelp_restaurant", data_root)
        self._content = {
            "num_classes": 11,
            "num_vertices": 50758,
            "num_edges": 679302,
            "dim_features": 1862,
            "features": {
                "upon": [
                    {
                        "filename": "features.pkl",
                        "md5": "cedc4443884477c2e626025411c44cd7",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [
                    to_tensor,
                ],
            },
            "edge_list": {
                "upon": [
                    {
                        "filename": "edge_list.pkl",
                        "md5": "4b26eecaa22305dd10edcd6372eb49da",
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "1cdc1ed9fb1f57b2accaa42db214d4ef",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "state": {
                "upon": [
                    {"filename": "state.pkl", "md5": "eef3b835fad37409f29ad36539296b57"}
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "city": {
                "upon": [
                    {"filename": "city.pkl", "md5": "8302b167262b23067698e865cacd0b17"}
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
