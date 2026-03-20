#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from contextlib import contextmanager

from .dl_import import *
dl_import("from . import libgraph_tool_core as libcore")

def openmp_enabled():
    """Return ``True`` if OpenMP was enabled during compilation."""
    return libcore.openmp_enabled()


_hooks = []


def _openmp_add_hook(hook):
    _hooks.append(hook)


def openmp_get_num_threads():
    """Return the number of OpenMP threads."""
    return libcore.openmp_get_num_threads()


def openmp_set_num_threads(n):
    """Set the number of OpenMP threads."""
    libcore.openmp_set_num_threads(n)
    for hook in _hooks:
        hook()


def openmp_get_schedule():
    """Return the runtime OpenMP schedule and chunk size. The schedule can by
    any of: ``"static"``, ``"dynamic"``, ``"guided"``, ``"auto"``."""
    return libcore.openmp_get_schedule()


def openmp_set_schedule(schedule, chunk=0):
    """Set the runtime OpenMP schedule and chunk size. The schedule can by
    any of: ``"static"``, ``"dynamic"``, ``"guided"``, ``"auto"``."""
    return libcore.openmp_set_schedule(schedule, chunk)


def openmp_get_thresh():
    """Return the minimum number of vertices necessary to enable parallelization."""
    return libcore.openmp_get_thresh()


def openmp_set_thresh(n):
    """Set the the minimum number of vertices necessary to enable parallelization."""
    return libcore.openmp_set_thresh(n)


@contextmanager
def openmp_context(nthreads=None, schedule=None, chunk=0, thresh=None):
    """Return a context manager that sets the tuntime OpenMP parameters, and
    restores the original values when exited."""
    nthreads_ = openmp_get_num_threads()
    schedule_, chunk_ = openmp_get_schedule()
    thres_ = openmp_get_thresh()
    try:
        if nthreads is not None:
            openmp_set_num_threads(nthreads)
        if schedule is not None:
            openmp_set_schedule(schedule, chunk)
        if thresh is not None:
            openmp_set_thresh(thresh)
        yield
    finally:
        openmp_set_num_threads(nthreads_)
        openmp_set_schedule(schedule_, chunk_)
        openmp_set_thresh(thres_)
