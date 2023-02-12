import easygraph.classes
import easygraph.convert
import easygraph.datasets
import easygraph.functions
import easygraph.model
import easygraph.nn
import easygraph.random
import easygraph.readwrite
import easygraph.utils

from easygraph.classes import *
from easygraph.convert import *
from easygraph.datasets import *
from easygraph.functions import *
from easygraph.model import *
from easygraph.nn import *
from easygraph.random import *
from easygraph.readwrite import *
from easygraph.utils import *


def __getattr__(name):
    print(f"attr {name} doesn't exist!")
