try:
    from easygraph.datasets.get_sample_graph import *
    from easygraph.datasets.gnn_benchmark import *
    from easygraph.datasets.hypergraph.dynamic.email_enron import Email_Enron
    from easygraph.datasets.hypergraph.dynamic.email_eu import Email_Eu
    from easygraph.datasets.hypergraph.dynamic.hospital_lyon import Hospital_Lyon
    from easygraph.datasets.karate import KarateClubDataset

    from .citation_graph import CitationGraphDataset
    from .citation_graph import CiteseerGraphDataset
    from .citation_graph import CoraBinary
    from .citation_graph import CoraGraphDataset
    from .citation_graph import PubmedGraphDataset
    from .cooking_200 import Cooking200
    from .hypergraph import *
    from .ppi import LegacyPPIDataset
    from .ppi import PPIDataset

except:
    print(
        " Please install Pytorch before use dataset such as"
        " KarateClubDataset、CitationDataset、PPIDataset、LegacyPPIDataset"
    )
