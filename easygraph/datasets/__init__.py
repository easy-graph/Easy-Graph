# risky imports
try:
    from easygraph.datasets.get_sample_graph import *
    from easygraph.datasets.gnn_benchmark import *
    from easygraph.datasets.hypergraph.coauthorship import *
    from easygraph.datasets.hypergraph.contact_primary_school import *
    from easygraph.datasets.hypergraph.cooking_200 import Cooking200
    from easygraph.datasets.hypergraph.House_Committees import House_Committees
    from easygraph.datasets.karate import KarateClubDataset
    from easygraph.datasets.mathoverflow_answers import mathoverflow_answers
    from .ppi import LegacyPPIDataset
    from .ppi import PPIDataset
except Exception as e:
    print(
        " Please install Pytorch before use graph-related datasets and"
        " hypergraph-related datasets."
    )

from .citation_graph import (
    CitationGraphDataset,
    CiteseerGraphDataset,
    CoraBinary,
    CoraGraphDataset,
    PubmedGraphDataset,
)
from .coauthor import CoauthorCSDataset
from .amazon_photo import AmazonPhotoDataset
from .reddit import RedditDataset
from .flickr import FlickrDataset
from .facebook_ego import FacebookEgoNetDataset
from .roadnet import RoadNetCADataset
from .arxiv import ArxivHEPTHDataset
from .github import GitHubUsersDataset
from .twitter_ego import TwitterEgoDataset
from .wiki_topcats import WikiTopCatsDataset