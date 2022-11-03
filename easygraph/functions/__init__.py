from easygraph.convert import *
from easygraph.functions.basic import *
from easygraph.functions.centrality import *
from easygraph.functions.community import *
from easygraph.functions.components import *
from easygraph.functions.drawing import *
from easygraph.functions.graph_embedding import *
from easygraph.functions.graph_generator import *
from easygraph.functions.isolate import *
from easygraph.functions.path import *
from easygraph.functions.structural_holes import *


def __getattr__(name):
    print(f"atte {name} doesn't exist!")
