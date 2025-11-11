"""
EasyGraph Capabilities Module

This module provides information about what EasyGraph can do.
It helps users discover the available features and functionalities.
"""


def show_capabilities():
    """
    Display comprehensive information about EasyGraph's capabilities.

    This function prints a structured overview of what EasyGraph can do,
    including available features, algorithms, and modules.

    Usage:
        >>> import easygraph as eg
        >>> eg.show_capabilities()

    Returns:
        None: Prints the capabilities to stdout.
    """
    capabilities_text = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    EasyGraph Capabilities (你能做什么？)                   ║
║                         What Can EasyGraph Do?                           ║
╚══════════════════════════════════════════════════════════════════════════╝

EasyGraph is a comprehensive network analysis library. Here's what it can do:

📊 GRAPH CREATION & MANIPULATION
  • Create graphs: Graph(), DiGraph(), MultiGraph(), MultiDiGraph()
  • Add/remove nodes and edges
  • Graph conversion between different types
  • Support for various input formats (edge lists, adjacency matrices, etc.)

📈 CENTRALITY MEASURES
  • Degree centrality
  • Betweenness centrality
  • Closeness centrality
  • PageRank
  • Katz centrality
  • Ego betweenness
  • Flow betweenness
  • Laplacian centrality

🔍 COMMUNITY DETECTION
  • Louvain algorithm
  • Label Propagation Algorithm (LPA)
  • Modularity-based detection
  • Ego graph extraction
  • Motif detection

🕳️ STRUCTURAL HOLE ANALYSIS
  • HIS (Structural Hole Information Diffusion)
  • HAM (Hierarchical Affiliation Model)
  • MaxD (Maximum Degree)
  • AP_Greedy
  • Constraint metrics
  • Effective size
  • Various structural hole evaluation metrics

🌐 NETWORK COMPONENTS
  • Connected components
  • Strongly connected components (directed)
  • Weakly connected components (directed)
  • Biconnected components

🧮 BASIC NETWORK METRICS
  • Clustering coefficient
  • Average degree
  • Local assortativity
  • Diameter
  • Average shortest path length

🛤️ PATH ALGORITHMS
  • Shortest paths (single-source and all-pairs)
  • Bridges detection
  • Minimum spanning tree (MST)
  • Dijkstra's algorithm

🎯 CORE DECOMPOSITION
  • K-core decomposition
  • Core number calculation

📊 GRAPH EMBEDDING
  • DeepWalk
  • Node2Vec
  • LINE (Large-scale Information Network Embedding)
  • SDNE (Structural Deep Network Embedding)
  • NOBE

🎲 GRAPH GENERATION
  • Random networks (Erdős-Rényi, Barabási-Albert, etc.)
  • Classic graphs (complete, cycle, path, star, etc.)
  • Network generators for various models

🔺 HYPERGRAPH ANALYSIS
  • Hypergraph creation and manipulation
  • Hypergraph clustering
  • Hypergraph centrality measures
  • Hypergraph assortativity
  • Various hypergraph operations

⚡ GPU ACCELERATION (EGGPU)
  • GPU-accelerated betweenness centrality
  • K-core centrality on GPU
  • Single-source shortest path on GPU
  • Structural hole metrics on GPU
  • Significant speedup for large-scale networks

🎨 VISUALIZATION
  • Network drawing and layout
  • Dynamic network visualization
  • Hypergraph visualization
  • Various layout algorithms (spring, circular, hierarchical, etc.)

📚 DATASETS
  • Built-in network datasets
  • Easy dataset loading
  • Support for various network data formats

🤖 MACHINE LEARNING
  • Graph neural networks (GNN)
  • Network embedding methods
  • ML metrics for graph tasks

📖 I/O OPERATIONS
  • Read/write various graph formats
  • Edge list, adjacency list, GML, GraphML
  • Custom format support

════════════════════════════════════════════════════════════════════════════

📦 INSTALLATION:
  pip install --upgrade Python-EasyGraph

📚 DOCUMENTATION:
  https://easy-graph.github.io/

💻 SOURCE CODE:
  https://github.com/easy-graph/Easy-Graph

🐛 ISSUES & QUESTIONS:
  https://github.com/easy-graph/Easy-Graph/issues

🎥 YOUTUBE CHANNEL:
  https://www.youtube.com/@python-easygraph

════════════════════════════════════════════════════════════════════════════

📋 QUICK START EXAMPLES:

1. Basic Graph Creation:
   >>> import easygraph as eg
   >>> G = eg.Graph()
   >>> G.add_edges([(1,2), (2,3), (1,3)])

2. PageRank Calculation:
   >>> eg.pagerank(G)

3. Community Detection:
   >>> communities = eg.louvain(G)

4. Structural Hole Detection:
   >>> _, _, H = eg.get_structural_holes_HIS(G, C=[frozenset([1,2,3])])

5. Network Embedding:
   >>> model = eg.DeepWalk(G, dimensions=128)
   >>> embeddings = model.train()

For more examples, visit: https://easy-graph.github.io/

════════════════════════════════════════════════════════════════════════════
"""
    print(capabilities_text)


def get_capabilities_dict():
    """
    Get a dictionary containing EasyGraph's capabilities organized by category.

    This function returns a structured dictionary that can be used
    programmatically to access information about EasyGraph's features.

    Returns:
        dict: A dictionary with categories as keys and lists of features as values.

    Example:
        >>> import easygraph as eg
        >>> caps = eg.get_capabilities_dict()
        >>> print(caps['centrality'])
    """
    capabilities = {
        "graph_types": [
            "Graph (undirected)",
            "DiGraph (directed)",
            "MultiGraph (undirected with parallel edges)",
            "MultiDiGraph (directed with parallel edges)",
        ],
        "centrality": [
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "pagerank",
            "katz_centrality",
            "ego_betweenness",
            "flow_betweenness",
            "laplacian_centrality",
        ],
        "community_detection": [
            "louvain",
            "LPA (Label Propagation Algorithm)",
            "modularity_based_detection",
            "ego_graph",
            "motif_detection",
        ],
        "structural_holes": [
            "get_structural_holes_HIS",
            "get_structural_holes_HAM",
            "get_structural_holes_MaxD",
            "AP_Greedy",
            "constraint",
            "effective_size",
            "ICC (Information Centrality Constraint)",
        ],
        "components": [
            "connected_components",
            "strongly_connected_components",
            "weakly_connected_components",
            "biconnected_components",
        ],
        "basic_metrics": [
            "clustering_coefficient",
            "average_degree",
            "local_assortativity",
            "diameter",
            "average_shortest_path_length",
        ],
        "path_algorithms": [
            "shortest_path",
            "all_pairs_shortest_path",
            "dijkstra",
            "bridges",
            "minimum_spanning_tree",
        ],
        "core_decomposition": [
            "k_core",
            "core_number",
        ],
        "graph_embedding": [
            "DeepWalk",
            "Node2Vec",
            "LINE",
            "SDNE",
            "NOBE",
        ],
        "graph_generation": [
            "erdos_renyi_graph",
            "barabasi_albert_graph",
            "complete_graph",
            "cycle_graph",
            "path_graph",
            "star_graph",
        ],
        "hypergraph": [
            "Hypergraph class",
            "hypergraph_clustering",
            "hypergraph_centrality",
            "hypergraph_assortativity",
        ],
        "gpu_acceleration": [
            "GPU betweenness centrality",
            "GPU k-core",
            "GPU shortest path",
            "GPU structural holes",
        ],
        "visualization": [
            "draw",
            "draw_spring",
            "draw_circular",
            "dynamic_visualization",
            "hypergraph_visualization",
        ],
        "io_formats": [
            "edge_list",
            "adjacency_list",
            "GML",
            "GraphML",
            "custom_formats",
        ],
    }
    return capabilities


# Alias for Chinese users
能做什么 = show_capabilities


__all__ = [
    "show_capabilities",
    "get_capabilities_dict",
    "能做什么",
]
