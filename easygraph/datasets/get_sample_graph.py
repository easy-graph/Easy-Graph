import sys
sys.path.append('../../')
import easygraph as eg

__all__ = [
    'get_graph_blogcatalog',
    'get_graph_youtube',
    'get_graph_flickr'
]


def get_graph_blogcatalog():
    """
    Returns the undirected graph of blogcatalog.
    Dataset from:
    https://github.com/phanein/deepwalk/blob/master/example_graphs/blogcatalog.mat

    References
    ----------
    .. [1] Bryan Perozzi, Rami Al-Rfou, Steven Skiena. KDD'2014
       DeepWalk: Online Learning of Social Representations
    """
    from scipy.io import loadmat
    def sparse2graph(x):
        from collections import defaultdict
        from six import iteritems

        G = defaultdict(lambda: set())
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].add(j)
        return {str(k): [str(x) for x in v] for k, v in iteritems(G)}
    
    mat = loadmat('./samples/blogcatalog.mat')
    data = sparse2graph(A)

    G = eg.Graph()
    for u in data:
        for v in data[u]:
            G.add_edge(u, v)
    
    return G


def get_graph_youtube():
    """
    Returns the undirected graph of youtube dataset.
    Dataset from:
    http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz
    """
    from urllib import request
    import gzip
    url = 'http://socialnetworks.mpi-sws.mpg.de/data/youtube-links.txt.gz'
    zipped_data_path = './samples/youtube-links.txt.gz'
    unzipped_data_path = './samples/youtube-links.txt'

    # Download .gz file
    request.urlretrieve(url, zipped_data_path)

    # Unzip
    unzipped_data = gzip.GzipFile(zipped_data_path)
    open(unzipped_data_path, 'wb+').write(unzipped_data.read())
    unzipped_data.close()

    # Returns graph
    G = eg.Graph()
    G.add_edges_from_file(file=unzipped_data_path)
    return G


def get_graph_flickr():
    """
    Returns the undirected graph of youtube dataset.
    Dataset from:
    http://socialnetworks.mpi-sws.mpg.de/data/flickr-links.txt.gz
    """
    from urllib import request
    import gzip
    url = 'http://socialnetworks.mpi-sws.mpg.de/data/flickr-links.txt.gz'
    zipped_data_path = './samples/flickr-links.txt.gz'
    unzipped_data_path = './samples/flickr-links.txt'

    # Download .gz file
    request.urlretrieve(url, zipped_data_path)

    # Unzip
    unzipped_data = gzip.GzipFile(zipped_data_path)
    open(unzipped_data_path, 'wb+').write(unzipped_data.read())
    unzipped_data.close()

    # Returns graph
    G = eg.Graph()
    G.add_edges_from_file(file=unzipped_data_path)
    return G
