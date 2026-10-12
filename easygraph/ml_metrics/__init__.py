from typing import Dict
from typing import List
from typing import Union

from easygraph._global import AUTHOR_EMAIL
from easygraph._optional import HAS_TORCH as _HAS_TORCH
from easygraph._optional import optional_torch_modules as _torch_imports
from easygraph._optional import require_torch as _require_torch


if _HAS_TORCH:
    with _torch_imports():
        from .base import BaseEvaluator
        from .classification import VertexClassificationEvaluator
        from .classification import available_classification_metrics
        from .hypergraphs import HypergraphVertexClassificationEvaluator


def build_evaluator(
    task: str,
    metric_configs: List[Union[str, Dict[str, dict]]],
    validate_index: int = 0,
):
    r"""Return the metric evaluator for the given task.

    Args:
        ``task`` (``str``): The type of the task. The supported types include: ``graph_vertex_classification``, ``hypergraph_vertex_classification``, and ``user_item_recommender``.
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The list of metric names.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.
    """
    _require_torch("Machine-learning evaluators")
    if task == "hypergraph_vertex_classification":
        return HypergraphVertexClassificationEvaluator(metric_configs, validate_index)
    else:
        raise ValueError(
            f"{task} is not supported yet. Please email '{AUTHOR_EMAIL}' to add it."
        )


# __all__ = [
#     "BaseEvaluator",
#     "build_evaluator",
#     "available_classification_metrics",
#     "VertexClassificationEvaluator",
#     "HypergraphVertexClassificationEvaluator",
# ]
