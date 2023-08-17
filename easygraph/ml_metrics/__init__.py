try:
    from typing import Dict
    from typing import List
    from typing import Union

    from easygraph._global import AUTHOR_EMAIL

    from .base import BaseEvaluator
    from .classification import VertexClassificationEvaluator
    from .classification import available_classification_metrics
    from .hypergraphs import HypergraphVertexClassificationEvaluator
except:
    print(
        "Warning raise in module:ml_metrics. Please install Pytorch before you use"
        " functions related to nueral network"
    )


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
