import easygraph.classes
import easygraph.convert
import easygraph.datapipe
import easygraph.datasets
import easygraph.exception
import easygraph.experiments
import easygraph.functions
import easygraph.ml_metrics
import easygraph.model
import easygraph.nn
import easygraph.readwrite
import easygraph.utils

from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import require_torch as _require_torch
from easygraph._optional import warn_missing_torch as _warn_missing_torch
from easygraph.classes import *
from easygraph.convert import *
from easygraph.datapipe import *
from easygraph.datasets import *
from easygraph.exception import *
from easygraph.experiments import *
from easygraph.functions import *
from easygraph.ml_metrics import *
from easygraph.model import *
from easygraph.nn import *
from easygraph.readwrite import *
from easygraph.utils import *


_warn_missing_torch()


def __getattr__(name):
    if not _HAS_TORCH and name in {"Hypergraph", "BaseHypergraph", "load_structure"}:
        _require_torch(name)
    raise AttributeError("module 'easygraph' has no attribute {!r}".format(name))
