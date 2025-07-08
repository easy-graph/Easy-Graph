import json
import os

from easygraph.classes.hypergraph import Hypergraph
from easygraph.datasets.dynamic.load_dataset import request_json_from_url
from easygraph.datasets.graph_dataset_base import EasyGraphDataset
from easygraph.datasets.utils import _get_eg_url
from easygraph.datasets.utils import tensor


class Hospital_Lyon(EasyGraphDataset):
    _urls = {
        "hospital_lyon": (
            "easygraph-data-hospital-lyon/-/raw/main/hospital-lyon.json?ref_type=heads&inline=false"
        ),
    }

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
        save_dir="./",
    ):
        name = "hospital_lyon"
        self.url = _get_eg_url(self._urls[name])
        super(Hospital_Lyon, self).__init__(
            name=name,
            url=self.url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
            save_dir=save_dir,
        )

    def preprocess(self, data, max_order=None, is_dynamic=True):
        # The index of the nodes in this dataset are not continuous and therefore require special processing
        timestamp_lst = list()
        node_data = data["node-data"]
        node_num = len(node_data)
        G = Hypergraph(num_v=node_num)
        id = 0
        name_dict = {}
        for k, v in data["node-data"].items():
            name_dict[k] = id
            v["name"] = k
            G.v_property[id] = v
            id = id + 1
        e_property_dict = data["edge-data"]
        rows = []
        cols = []
        edge_flag_dict = {}
        edge_id = 0
        for id, edge in data["edge-dict"].items():
            if max_order and len(edge) > max_order + 1:
                continue

            try:
                id = int(id)
            except ValueError as e:
                raise TypeError(
                    f"Failed to convert the edge with ID {id} to type int."
                ) from e

            try:
                edge = [name_dict[n] for n in edge]
                rows.extend(edge)
                cols.extend(len(edge) * [edge_id])
                edge_id += 1
            except ValueError as e:
                raise TypeError(f"Failed to convert nodes to type int.") from e
            if is_dynamic:
                G.add_hyperedges(
                    e_list=edge,
                    e_property=e_property_dict[str(id)],
                    group_name=e_property_dict[str(id)]["timestamp"],
                )
                timestamp_lst.append(e_property_dict[str(id)]["timestamp"])
            else:
                G.add_hyperedges(e_list=edge, e_property=e_property_dict[str(id)])
        G._rows = rows
        G._cols = cols
        return G, timestamp_lst

    @property
    def url(self):
        return self._url

    @property
    def save_name(self):
        return self.name

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def load(self):
        graph_path = os.path.join(self.save_path, self.save_name + ".json")
        with open(graph_path, "r") as f:
            self.load_data = json.load(f)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.save_name + ".json")
        if os.path.exists(graph_path):
            return True
        return False

    def download(self):
        if self.has_cache():
            self.load()
        else:
            root = self.raw_dir
            data = request_json_from_url(self.url)
            with open(os.path.join(root, self.save_name + ".json"), "w") as f:
                json.dump(data, f)
            self.load_data = data

    def process(self):
        """Loads input data from data directory and transfer to target graph for better analysis"""

        self._g, edge_feature_list = self.preprocess(self.load_data, is_dynamic=True)
        self._g.ndata["hyperedge_feature"] = tensor(
            range(1, len(edge_feature_list) + 1)
        )

    @url.setter
    def url(self, value):
        self._url = value
