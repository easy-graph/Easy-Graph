import easygraph as eg


# import progressbar


__all__ = [
    "get_graph_karateclub",
    "get_graph_blogcatalog",
    "get_graph_youtube",
    "get_graph_flickr",
]


def get_graph_karateclub():
    """Returns the undirected graph of Karate Club.

    Returns
    -------
    get_graph_karateclub : easygraph.Graph
        The undirected graph instance of karate club from dataset:
        http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm

    References
    ----------
    .. [1] http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm

    """
    all_members = set(range(34))
    club1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21}
    # club2 = all_members - club1

    G = eg.Graph(name="Zachary's Karate Club")
    for node in all_members:
        G.add_node(node + 1)

    zacharydat = """\
0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0
1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0
1 1 0 1 0 0 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0
1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1
0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1
0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 1 1 1 0 1
0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 0"""

    for row, line in enumerate(zacharydat.split("\n")):
        thisrow = [int(b) for b in line.split()]
        for col, entry in enumerate(thisrow):
            if entry == 1:
                G.add_edge(row + 1, col + 1)

    # Add the name of each member's club as a node attribute.
    for v in G:
        G.nodes[v]["club"] = "Mr. Hi" if v in club1 else "Officer"
    return G


def get_graph_blogcatalog():
    """Returns the undirected graph of blogcatalog.

    Returns
    -------
    get_graph_blogcatalog : easygraph.Graph
        The undirected graph instance of blogcatalog from dataset:
        https://github.com/phanein/deepwalk/blob/master/example_graphs/blogcatalog.mat

    References
    ----------
    .. [1] https://github.com/phanein/deepwalk/blob/master/example_graphs/blogcatalog.mat

    """
    from scipy.io import loadmat

    def sparse2graph(x):
        from collections import defaultdict

        G = defaultdict(lambda: set())
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].add(j)
        return {str(k): [str(x) for x in v] for k, v in G.items()}

    mat = loadmat("./samples/blogcatalog.mat")
    A = mat["network"]
    data = sparse2graph(A)

    G = eg.Graph()
    for u in data:
        for v in data[u]:
            G.add_edge(u, v)

    return G


def get_graph_youtube():
    """Returns the undirected graph of Youtube dataset.

    Returns
    -------
    get_graph_youtube : easygraph.Graph
        The undirected graph instance of Youtube from dataset:
        http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz

    References
    ----------
    .. [1] http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz

    """
    import gzip

    from urllib import request

    url = "http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz"
    zipped_data_path = "./samples/youtube-links.txt.gz"
    unzipped_data_path = "./samples/youtube-links.txt"

    # Download .gz file
    print("Downloading Youtube dataset...")
    request.urlretrieve(url, zipped_data_path, _show_progress)

    # Unzip
    unzipped_data = gzip.GzipFile(zipped_data_path)
    open(unzipped_data_path, "wb+").write(unzipped_data.read())
    unzipped_data.close()

    # Returns graph
    G = eg.Graph()
    G.add_edges_from_file(file=unzipped_data_path)
    return G


def get_graph_flickr():
    """Returns the undirected graph of Flickr dataset.

    Returns
    -------
    get_graph_flickr : easygraph.Graph
        The undirected graph instance of Flickr from dataset:
        http://socialnetworks.mpi-sws.mpg.de/data/flickr-links.txt.gz

    References
    ----------
    .. [1] http://socialnetworks.mpi-sws.mpg.de/data/flickr-links.txt.gz

    """
    import gzip

    from urllib import request

    url = "http://socialnetworks.mpi-sws.mpg.de/data/flickr-links.txt.gz"
    zipped_data_path = "./samples/flickr-links.txt.gz"
    unzipped_data_path = "./samples/flickr-links.txt"

    # Download .gz file
    print("Downloading Flickr dataset...")
    request.urlretrieve(url, zipped_data_path, _show_progress)

    # Unzip
    unzipped_data = gzip.GzipFile(zipped_data_path)
    open(unzipped_data_path, "wb+").write(unzipped_data.read())
    unzipped_data.close()

    # Returns graph
    G = eg.Graph()
    G.add_edges_from_file(file=unzipped_data_path)
    return G


_pbar = None


def _show_progress(block_num, block_size, total_size):
    global _pbar
    if _pbar is None:
        _pbar = progressbar.ProgressBar(maxval=total_size)
        _pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        _pbar.update(downloaded)
    else:
        _pbar.finish()
        _pbar = None
