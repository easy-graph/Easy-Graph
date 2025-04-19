import requests

from easygraph.utils.exception import EasyGraphError


def request_text_from_url(url):
    """
    Requests text data from the specified URL.

    Args:
        url (str): The URL from which to request the text data.

    Returns:
        str: The text content of the response if the request is successful.

    Raises:
        EasyGraphError: If a connection error occurs during the request or if the HTTP response status code
                        indicates a failure.
    """
    try:
        r = requests.get(url)
    except requests.ConnectionError:
        raise EasyGraphError("Connection Error!")

    if r.ok:
        return r.text
    else:
        raise EasyGraphError(f"Error: HTTP response {r.status_code}")


class House_Committees:
    """
    A class for loading and processing the House Committees hypergraph dataset.

    This class fetches hyperedge, node label, node name, and label name data from predefined URLs,
    processes the data, and generates a hypergraph representation. It also provides access to various
    dataset attributes through properties and indexing.

    Attributes:
        data_root (str): The root URL for the data. If `data_root` is provided during initialization,
                         it is set to "https://"; otherwise, it is `None`.
        hyperedges_path (str): The URL of the file containing hyperedge information.
        node_labels_path (str): The URL of the file containing node label information.
        node_names_path (str): The URL of the file containing node name information.
        label_names_path (str): The URL of the file containing label name information.
        _hyperedges (list): A list of tuples representing hyperedges.
        _node_labels (list): A list of node labels.
        _label_names (list): A list of label names.
        _node_names (list): A list of node names.
        _content (dict): A dictionary containing dataset statistics and data, including the number of
                         classes, vertices, edges, the edge list, and node labels.
    """

    def __init__(self, data_root=None):
        """
        Initializes a new instance of the `House_Committees` class.

        Args:
            data_root (str, optional): The root URL for the data. If provided, it is set to "https://";
                                       otherwise, it is `None`. Defaults to `None`.
        """
        self.data_root = "https://" if data_root is not None else data_root
        self.hyperedges_path = "https://gitlab.com/easy-graph/easygraph-data-house-committees/-/raw/main/hyperedges-house-committees.txt?inline=false"
        self.node_labels_path = "https://gitlab.com/easy-graph/easygraph-data-house-committees/-/raw/main/node-labels-house-committees.txt?ref_type=heads&inline=false"
        self.node_names_path = "https://gitlab.com/easy-graph/easygraph-data-house-committees/-/raw/main/node-names-house-committees.txt?ref_type=heads&inline=false"
        self.label_names_path = "https://gitlab.com/easy-graph/easygraph-data-house-committees/-/raw/main/label-names-house-committees.txt?ref_type=heads&inline=false"
        self._hyperedges = []
        self._node_labels = []
        self._label_names = []
        self._node_names = []
        self.generate_hypergraph(
            hyperedges_path=self.hyperedges_path,
            node_labels_path=self.node_labels_path,
            node_names_path=self.node_names_path,
            label_names_path=self.label_names_path,
        )

        self._content = {
            "num_classes": len(self._label_names),
            "num_vertices": len(self._node_labels),
            "num_edges": len(self._hyperedges),
            "edge_list": self._hyperedges,
            "labels": self._node_labels,
        }

    def process_label_txt(self, data_str, delimiter="\n", transform_fun=str):
        """
        Processes a string containing label data into a list of transformed values.

        Args:
            data_str (str): The input string containing label data.
            delimiter (str, optional): The delimiter used to split the input string. Defaults to "\n".
            transform_fun (callable, optional): A function used to transform each label value.
                                                Defaults to the `str` function.

        Returns:
            list: A list of transformed label values.
        """
        data_str = data_str.strip()
        data_lst = data_str.split(delimiter)
        final_lst = []
        for data in data_lst:
            data = data.strip()
            data = transform_fun(data)
            final_lst.append(data)
        return final_lst

    def __getitem__(self, key: str):
        """
        Retrieves a value from the `_content` dictionary using the specified key.

        Args:
            key (str): The key used to access the `_content` dictionary.

        Returns:
            Any: The value corresponding to the key in the `_content` dictionary.
        """
        return self._content[key]

    @property
    def node_labels(self):
        """
        Gets the list of node labels.

        Returns:
            list: A list of node labels.
        """
        return self._node_labels

    @property
    def node_names(self):
        """
        Gets the list of node names.

        Returns:
            list: A list of node names.
        """
        return self._node_names

    @property
    def label_names(self):
        """
        Gets the list of label names.

        Returns:
            list: A list of label names.
        """
        return self._label_names

    @property
    def hyperedges(self):
        """
        Gets the list of hyperedges.

        Returns:
            list: A list of tuples representing hyperedges.
        """
        return self._hyperedges

    def generate_hypergraph(
        self,
        hyperedges_path=None,
        node_labels_path=None,
        node_names_path=None,
        label_names_path=None,
    ):
        """
        Generates a hypergraph by fetching and processing data from the specified URLs.

        Args:
            hyperedges_path (str, optional): The URL of the file containing hyperedge information.
                                             Defaults to `None`.
            node_labels_path (str, optional): The URL of the file containing node label information.
                                              Defaults to `None`.
            node_names_path (str, optional): The URL of the file containing node name information.
                                             Defaults to `None`.
            label_names_path (str, optional): The URL of the file containing label name information.
                                              Defaults to `None`.
        """

        def fun(data):
            """
            Converts a string to an integer and subtracts 1.

            Args:
                data (str): The input string to be converted.

            Returns:
                int: The converted integer value minus 1.
            """
            data = int(data) - 1
            return data

        hyperedges_info = request_text_from_url(hyperedges_path)
        hyperedges_info = hyperedges_info.strip()
        hyperedges_lst = hyperedges_info.split("\n")
        for hyperedge in hyperedges_lst:
            hyperedge = hyperedge.strip()
            hyperedge = [int(i) - 1 for i in hyperedge.split(",")]
            self._hyperedges.append(tuple(hyperedge))
        # print(self.hyperedges)

        node_labels_info = request_text_from_url(node_labels_path)

        process_node_labels_info = self.process_label_txt(
            node_labels_info, transform_fun=fun
        )
        self._node_labels = process_node_labels_info
        # print("process_node_labels_info:", process_node_labels_info)
        node_names_info = request_text_from_url(node_names_path)
        process_node_names_info = self.process_label_txt(node_names_info)
        self._node_names = process_node_names_info
        # print("process_node_names_info:", process_node_names_info)
        label_names_info = request_text_from_url(label_names_path)
        process_label_names_info = self.process_label_txt(label_names_info)
        self._label_names = process_label_names_info
