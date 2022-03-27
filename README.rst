EasyGraph
==================

Copyright (C) <2020-2022> by Mobile Systems and Networking Group, Fudan University

.. image:: https://img.shields.io/pypi/v/Python-EasyGraph.svg
  :target: https://pypi.org/project/Python-EasyGraph/
  
.. image:: https://img.shields.io/pypi/pyversions/Python-EasyGraph.svg
   :target: https://pypi.org/project/Python-EasyGraph/
   
.. image:: https://img.shields.io/pypi/l/Python-EasyGraph
   :target: https://github.com/easy-graph/Easy-Graph/blob/master/LICENSE
   
- **Documentation:** https://easy-graph.github.io/
- **Source:** https://github.com/easy-graph/Easy-Graph
- **Bug Reports:** https://github.com/easy-graph/Easy-Graph/issues
- **PyPI Homepage:** https://pypi.org/project/Python-EasyGraph/

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

This is a simple example for the detection of `structural hole spanners <https://en.wikipedia.org/wiki/Structural_holes>`_ 
using the `HIS <https://keg.cs.tsinghua.edu.cn/jietang/publications/WWW13-Lou&Tang-Structural-Hole-Information-Diffusion.pdf>`_ algorithm.

.. code:: python

  >>> import easygraph as eg
  >>> G = eg.Graph()
  >>> G.add_edges([(1,2), (2,3), (1,3), (3,4), (4,5), (3,5), (5,6)])
  >>> _, _, H = eg.get_structural_holes_HIS(G, C=[frozenset([1,2,3]), frozenset([4,5,6])])
  >>> H # The structural hole score of each node. Note that node `4` is regarded as the most possible structural hole spanner.
  {1: {0: 0.703948974609375}, 
   2: {0: 0.703948974609375}, 
   3: {0: 1.2799804687499998}, 
   4: {0: 1.519976806640625}, 
   5: {0: 1.519976806640625}, 
   6: {0: 0.83595703125}
  }



