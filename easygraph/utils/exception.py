"""
**********
Exceptions
**********

Base exceptions and errors for EasyGraph.
"""

__all__ = ["EasyGraphException", "EasyGraphError", "EasyGraphNotImplemented"]


class EasyGraphException(Exception):
    """Base class for exceptions in EasyGraph."""


class EasyGraphError(EasyGraphException):
    """Exception for a serious error in EasyGraph"""


class EasyGraphNotImplemented(EasyGraphException):
    """Exception raised by algorithms not implemented for a type of graph."""
