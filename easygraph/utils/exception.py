"""
**********
Exceptions
**********

Base exceptions and errors for EasyGraph.
"""


__all__ = [
    "EasyGraphException",
    "EasyGraphError",
    "EasyGraphNotImplemented",
    "EasyGraphPointlessConcept",
]


class EasyGraphException(Exception):
    """Base class for exceptions in EasyGraph."""


class EasyGraphError(EasyGraphException):
    """Exception for a serious error in EasyGraph"""


class EasyGraphNotImplemented(EasyGraphException):
    """Exception raised by algorithms not implemented for a type of graph."""


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
