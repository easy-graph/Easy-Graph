import requests

from easygraph.utils.exception import EasyGraphError


def request_text_from_url(url):
    """Requests text data from the specified URL.

    Args:
        url (str): The URL from which to request data.

    Returns:
        str: The text content of the response if the request is successful.

    Raises:
        EasyGraphError: If a connection error occurs or the HTTP response status code indicates failure.
    """
    try:
        r = requests.get(url)
    except requests.ConnectionError:
        raise EasyGraphError("Connection Error!")

    if r.ok:
        return r.text
    else:
        raise EasyGraphError(f"Error: HTTP response {r.status_code}")


class contact_primary_school:
    """A class for loading and processing the primary school contact network hypergraph dataset.

    This class loads hyperedge, node label, and label name data from specified URLs and generates a hypergraph.

    Attributes:
        data_root (str): The root URL for the data. If not provided, it is set to None.
        hyperedges_path (str): The URL for the hyperedge data.
        node_labels_path (str): The URL for the node label data.
        label_names_path (str): The URL for the label name data.
        _hyperedges (list): A list storing hyperedges.
        _node_labels (list): A list storing node labels.
        _label_names (list): A list storing label names.
        _node_names (list): A list storing node names (currently unused).
        _content (dict): A dictionary containing dataset statistics and data.
    """

    def __init__(self, data_root=None):
        """Initializes an instance of the contact_primary_school class.

        Args:
            data_root (str, optional): The root URL for the data. Defaults to None.
        """
        self.data_root = "https://" if data_root is not None else data_root
        self.hyperedges_path = "https://gitlab.com/easy-graph/easygraph-data-contact-primary-school/-/raw/main/hyperedges-contact-primary-school.txt?inline=false"
        self.node_labels_path = "https://gitlab.com/easy-graph/easygraph-data-contact-primary-school/-/raw/main/node-labels-contact-primary-school.txt?ref_type=heads&inline=false"
        # self.node_names_path = "https://gitlab.com/easy-graph/easygraph-data-house-committees/-/raw/main/node-names-house-committees.txt?ref_type=heads&inline=false"
        self.label_names_path = "https://gitlab.com/easy-graph/easygraph-data-contact-primary-school/-/raw/main/label-names-contact-primary-school.txt?ref_type=heads&inline=false"
        self._hyperedges = []
        self._node_labels = []
        self._label_names = []
        self._node_names = []
        self.generate_hypergraph(
            hyperedges_path=self.hyperedges_path,
            node_labels_path=self.node_labels_path,
            # node_names_path=self.node_names_path,
            label_names_path=self.label_names_path,
        )
        self._content = {
            "num_classes": len(self._label_names),
            "num_vertices": len(self._node_labels),
            "num_edges": len(self._hyperedges),
            "edge_list": self._hyperedges,
            "labels": self._node_labels,
        }

    def __getitem__(self, key: str):
        """Accesses data in the _content dictionary by key.

        Args:
            key (str): The key of the data to access.

        Returns:
            Any: The value corresponding to the key in the _content dictionary.
        """
        return self._content[key]

    def process_label_txt(self, data_str, delimiter="\n", transform_fun=str):
        """Processes label data read from a text file.

        Args:
            data_str (str): A string containing label data.
            delimiter (str, optional): The delimiter used to split the string. Defaults to "\n".
            transform_fun (callable, optional): A function used to transform each label. Defaults to str.

        Returns:
            list: A list of processed labels.
        """
        data_str = data_str.strip()
        data_lst = data_str.split(delimiter)
        final_lst = []
        for data in data_lst:
            data = data.strip()
            data = transform_fun(data)
            final_lst.append(data)
        return final_lst

    @property
    def node_labels(self):
        """Gets the list of node labels.

        Returns:
            list: A list of node labels.
        """
        return self._node_labels

    """
    @property
    def node_names(self):
        return self._node_names
    """

    @property
    def label_names(self):
        """Gets the list of label names.

        Returns:
            list: A list of label names.
        """
        return self._label_names

    @property
    def hyperedges(self):
        """Gets the list of hyperedges.

        Returns:
            list: A list of hyperedges.
        """
        return self._hyperedges

    def generate_hypergraph(
        self,
        hyperedges_path=None,
        node_labels_path=None,
        # node_names_path=None,
        label_names_path=None,
    ):
        """Generates hypergraph data from specified URLs.

        Args:
            hyperedges_path (str, optional): The URL for the hyperedge data. Defaults to None.
            node_labels_path (str, optional): The URL for the node label data. Defaults to None.
            label_names_path (str, optional): The URL for the label name data. Defaults to None.
        """

        def fun(data):
            """Converts the input data to an integer and subtracts 1.

            Args:
                data (str): The input string data.

            Returns:
                int: The converted integer data.
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
        label_names_info = request_text_from_url(label_names_path)
        process_label_names_info = self.process_label_txt(label_names_info)
        self._label_names = process_label_names_info
