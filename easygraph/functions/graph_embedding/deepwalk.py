import random

from easygraph.functions.graph_embedding.node2vec import (
    _get_embedding_result_from_gensim_skipgram_model,
)
from easygraph.functions.graph_embedding.node2vec import learn_embeddings
from easygraph.utils import *
from tqdm import tqdm


__all__ = ["deepwalk"]


@not_implemented_for("multigraph")
def deepwalk(G, dimensions=128, walk_length=80, num_walks=10, **skip_gram_params):
    """Graph embedding via DeepWalk.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    dimensions : int
        Embedding dimensions, optional(default: 128)

    walk_length : int
        Number of nodes in each walk, optional(default: 80)

    num_walks : int
        Number of walks per node, optional(default: 10)

    skip_gram_params : dict
        Parameters for gensim.models.Word2Vec - do not supply `size`, it is taken from the `dimensions` parameter

    Returns
    -------
    embedding_vector : dict
        The embedding vector of each node

    most_similar_nodes_of_node : dict
        The most similar nodes of each node and its similarity

    Examples
    --------

    >>> deepwalk(G,
    ...          dimensions=128, # The graph embedding dimensions.
    ...          walk_length=80, # Walk length of each random walks.
    ...          num_walks=10, # Number of random walks.
    ...          skip_gram_params = dict( # The skip_gram parameters in Python package gensim.
    ...          window=10,
    ...             min_count=1,
    ...             batch_words=4,
    ...             iter=15
    ...          ))

    References
    ----------
    .. [1] https://arxiv.org/abs/1403.6652

    """
    G_index, index_of_node, node_of_index = G.to_index_node_graph()

    walks = simulate_walks(G_index, walk_length=walk_length, num_walks=num_walks)
    model = learn_embeddings(walks=walks, dimensions=dimensions, **skip_gram_params)

    (
        embedding_vector,
        most_similar_nodes_of_node,
    ) = _get_embedding_result_from_gensim_skipgram_model(
        G=G, index_of_node=index_of_node, node_of_index=node_of_index, model=model
    )

    del G_index
    return embedding_vector, most_similar_nodes_of_node


def simulate_walks(G, walk_length, num_walks):
    walks = []
    nodes = list(G.nodes)
    print("Walk iteration:")
    for walk_iter in tqdm(range(num_walks)):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(_deepwalk_walk(G, walk_length=walk_length, start_node=node))

    return walks


def _deepwalk_walk(G, walk_length, start_node):
    """
    Simulate a random walk starting from start node.
    """
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            pick_node = random.choice(cur_nbrs)
            walk.append(pick_node)
        else:
            break
    return walk
