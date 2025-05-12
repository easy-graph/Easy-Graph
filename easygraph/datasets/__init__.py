try:
    from easygraph.datasets.get_sample_graph import *
    from easygraph.datasets.gnn_benchmark import *
    from easygraph.datasets.hypergraph.coauthorship import *
    from easygraph.datasets.hypergraph.contact_primary_school import *
    from easygraph.datasets.hypergraph.cooking_200 import Cooking200
    from easygraph.datasets.hypergraph.House_Committees import House_Committees
    from easygraph.datasets.karate import KarateClubDataset
    from easygraph.datasets.mathoverflow_answers import mathoverflow_answers

    from .citation_graph import CitationGraphDataset
    from .citation_graph import CiteseerGraphDataset
    from .citation_graph import CoraBinary
    from .citation_graph import CoraGraphDataset
    from .citation_graph import PubmedGraphDataset
    from .ppi import LegacyPPIDataset
    from .ppi import PPIDataset

except:
    print(
        " Please install Pytorch before use graph-related datasets and"
        " hypergraph-related datasets."
    )
