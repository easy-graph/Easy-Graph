import easygraph.classes
import easygraph.datapipe
import easygraph.datasets
import easygraph.experiments
import easygraph.functions
import easygraph.ml_metrics
import easygraph.model
import easygraph.nn
import easygraph.random
import easygraph.readwrite
import easygraph.utils

from easygraph import convert
from easygraph.classes import *
from easygraph.convert import *
from easygraph.datapipe import *
from easygraph.datasets import *
from easygraph.experiments import *
from easygraph.functions import *
from easygraph.ml_metrics import *
from easygraph.model import *
from easygraph.nn import *
from easygraph.random import *
from easygraph.readwrite import *
from easygraph.utils import *


def __getattr__(name):
    print(f"attr {name} doesn'n exist!")
