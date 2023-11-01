import json
import os

from warnings import warn

import requests

from easygraph.convert import dict_to_hypergraph
from easygraph.utils.exception import EasyGraphError


__all__ = [
    "load_dynamic_hypergraph_dataset",
]

dataset_index_url = "https://gitlab.com/easy-graph/easygraph-data/-/raw/main/dataset_index.json?inline=false"


def request_json_from_url(url):
    try:
        r = requests.get(url)
    except requests.ConnectionError:
        raise EasyGraphError("Connection Error!")

    if r.ok:
        return r.json()
    else:
        raise EasyGraphError(f"Error: HTTP response {r.status_code}")


def _request_from_eg_data(dataset=None, cache=True):
    """Request a dataset from eg-data.

    Parameters
    ----------
    dataset : str, optional
        Dataset name. Valid options are the top-level tags of the
        index.json file in the xgi-data repository. If None, prints
        the list of available datasets.
    cache : bool, optional
        Whether or not to cache the output

    Returns
    -------
    Data
        The requested data loaded from a json file.

    Raises
    ------
    EasyGraphError
        If the HTTP request is not successful or the dataset does not exist.


    """

    index_data = request_json_from_url(dataset_index_url)

    key = dataset.lower()
    if key not in index_data:
        print("Valid dataset names:")
        print(*index_data, sep="\n")
        raise EasyGraphError("Must choose a valid dataset name!")

    return request_json_from_url(index_data[key]["url"])


def load_dynamic_hypergraph_dataset(
    dataset=None,
    local_read=False,
    path="",
    max_order=None,
):
    index_datasets = request_json_from_url(dataset_index_url)
    if dataset is None:
        print("Please refer to available list")

        print(*index_datasets, sep="\n")
        return

    if local_read:
        cfp = os.path.join(path, dataset + ".json")
        if os.path.exists(cfp):
            data = json.load(open(cfp, "r"))
            return dict_to_hypergraph(data, max_order=max_order)
        else:
            warn(
                f"No local copy was found at {cfp}. The data is requested "
                "from the xgi-data repository instead. To download a local "
                "copy, use `download_xgi_data`."
            )
    data = _request_from_eg_data(dataset)
    return dict_to_hypergraph(
        data, max_order=max_order, is_dynamic=index_datasets[dataset]["is_dynamic"]
    )
