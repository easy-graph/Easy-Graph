try:
    from easygraph.datasets.get_sample_graph import *
    from easygraph.datasets.gnn_benchmark import *
    from easygraph.datasets.karate import KarateClubDataset

    from .citation_graph import CitationGraphDataset
    from .citation_graph import CiteseerGraphDataset
    from .citation_graph import CoraBinary
    from .citation_graph import CoraGraphDataset
    from .citation_graph import PubmedGraphDataset
    from .cooking_200 import Cooking200
    from .email_enron import Email_Enron
    from .email_eu import Email_Eu
    from .hospital_lyon import Hospital_Lyon
    from .hypergraph import *
    from .ppi import LegacyPPIDataset
    from .ppi import PPIDataset

except:
    print(
        " Please install Pytorch before use dataset such as"
        " KarateClubDataset、CitationDataset、PPIDataset、LegacyPPIDataset"
    )
