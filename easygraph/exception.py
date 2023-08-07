"""
**********
Exceptions
**********

Base exceptions and errors for EasyGraph.
"""

__all__ = [
    "HasACycle",
    "NodeNotFound",
    "EasyGraphAlgorithmError",
    "EasyGraphException",
    "EasyGraphError",
    "EasyGraphNoCycle",
    "EasyGraphNoPath",
    "EasyGraphNotImplemented",
    "EasyGraphPointlessConcept",
    "EasyGraphUnbounded",
    "EasyGraphUnfeasible",
]


class EasyGraphException(Exception):
    """Base class for exceptions in EasyGraph."""


class EasyGraphError(EasyGraphException):
    """Exception for a serious error in EasyGraph"""


class EasyGraphPointlessConcept(EasyGraphException):
    """Raised when a null graph is provided as input to an algorithm
    that cannot use it.

    The null graph is sometimes considered a pointless concept [1]_,
    thus the name of the exception.

    References
    ----------
    .. [1] Harary, F. and Read, R. "Is the Null Graph a Pointless
       Concept?"  In Graphs and Combinatorics Conference, George
       Washington University.  New York: Springer-Verlag, 1973.

    """


class EasyGraphAlgorithmError(EasyGraphException):
    """Exception for unexpected termination of algorithms."""


class EasyGraphUnfeasible(EasyGraphAlgorithmError):
    """Exception raised by algorithms trying to solve a problem
    instance that has no feasible solution."""


class EasyGraphNoPath(EasyGraphUnfeasible):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist."""


class EasyGraphNoCycle(EasyGraphUnfeasible):
    """Exception for algorithms that should return a cycle when running
    on graphs where such a cycle does not exist."""


class HasACycle(EasyGraphException):
    """Raised if a graph has a cycle when an algorithm expects that it
    will have no cycles.

    """


class EasyGraphUnbounded(EasyGraphAlgorithmError):
    """Exception raised by algorithms trying to solve a maximization
    or a minimization problem instance that is unbounded."""


class EasyGraphNotImplemented(EasyGraphException):
    """Exception raised by algorithms not implemented for a type of graph."""


class NodeNotFound(EasyGraphException):
    """Exception raised if requested node is not present in the graph"""
