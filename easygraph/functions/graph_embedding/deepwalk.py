import sys
sys.path.append('../../../')
import easygraph as eg

from easygraph.functions.graph_embedding.node2vec import learn_embeddings
from easygraph.functions.graph_embedding.node2vec import _get_embedding_result_from_gensim_skipgram_model
import random
from tqdm import tqdm


__all__ = [
    "deepwalk"
]

def deepwalk(G, dimensions=128, walk_length=80, num_walks=10, **skip_gram_params):
    """
    Returns 
        1. The embedding vector of each node via DeepWalk: https://arxiv.org/abs/1403.6652
        2. The most similar nodes of each node and its similarity
    Using Word2Vec model of package gensim.

    Parameters
    ----------
    G : graph

    dimensions : int
        Embedding dimensions (default: 128)

    walk_length : int
        Number of nodes in each walk (default: 80)

    num_walks : int
        Number of walks per node (default: 10)

    skip_gram_params : dict
        Parameteres for gensim.models.Word2Vec - do not supply 'size', it is taken from the 'dimensions' parameter
    """
    G_index, index_of_node, node_of_index = G.to_index_node_graph()

    walks = simulate_walks(
        G_index, walk_length=walk_length, num_walks=num_walks)
    model = learn_embeddings(
        walks=walks, dimensions=dimensions, **skip_gram_params)

    embedding_vector, most_similar_nodes_of_node = _get_embedding_result_from_gensim_skipgram_model(
        G=G, index_of_node=index_of_node, node_of_index=node_of_index, model=model
    )
    
    del G_index
    return embedding_vector, most_similar_nodes_of_node


def simulate_walks(G, walk_length, num_walks):
    walks = []
    nodes = list(G.nodes)
    print('Walk iteration:')
    for walk_iter in tqdm(range(num_walks)):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(_deepwalk_walk(G,
                                        walk_length=walk_length,
                                        start_node=node))

    return walks

def _deepwalk_walk(G, walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''
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
