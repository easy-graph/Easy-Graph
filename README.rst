EasyGraph(v.0.2a4)
==================

Copyright (C) <2020> by Mobile Systems and Networking Group, Fudan University

.. image:: https://img.shields.io/pypi/v/Python-EasyGraph.svg
  :target: https://pypi.org/project/Python-EasyGraph/
  
.. image:: https://img.shields.io/pypi/pyversions/Python-EasyGraph.svg
   :target: https://pypi.org/project/Python-EasyGraph/
   
- **Documentation:** https://easy-graph.github.io/
- **Source:** https://github.com/easy-graph/Easy-Graph
- **Bug Reports:** https://github.com/easy-graph/Easy-Graph/issues

Introduction
------------
EasyGraph is an open source graph processing library. It is written in Python and supports analysis for undirected graphs and directed graphs. It covers advanced graph processing methods in structural hole spanners detection, graph embedding and several classic methods (subgraph generation, connected component discovery and isomorphic graph generation).

EasyGraph integrates state-of-the-art graph processing approaches and implements them using Python. EasyGraph covers a series of advanced graph processing algorithms include structural hole spanners detection (HIS, MaxD, Common_Greedy, AP_Greedy and HAM), and graph representation learning (DeepWalk, Node2Vec, LINE and SDNE). Besides, for a number of general graph processing approaches, EasyGraph optimizes the algorithms and introduces parallel computing methods to achieve high efficiency.

Install
-------
Installation with ``pip``
::

    $ pip install Python-EasyGraph
    
or ``conda``
::

    $ conda install Python-EasyGraph
    
Simple Example
--------------

This is a simple example of `sturctural hole spanners <https://en.wikipedia.org/wiki/Structural_holes>`_ detection with `MaxD <https://keg.cs.tsinghua.edu.cn/jietang/publications/WWW13-Lou&Tang-Structural-Hole-Information-Diffusion.pdf>`_.

.. code:: python
  >>> import easygraph as eg
  >>> G = eg.Graph()
  >>> G.add_edges([(1,2), (2,3), (1,3), (3,4), (4,5), (3,5), (5,6)])
  >>> M=eg.get_structural_holes_MaxD(G,
  ...                     k = 2, # To find top two structural holes spanners.
  ...                     C = [frozenset([1,2,3]), frozenset([4,5,6])] # Two communities
  ...                    )






