EasyGraph
==================

Copyright (C) <2020-2024> by DataNET Group, Fudan University

.. image:: https://img.shields.io/pypi/v/Python-EasyGraph.svg?label=PyPI
  :target: https://pypi.org/project/Python-EasyGraph/

.. image:: https://img.shields.io/pypi/pyversions/Python-EasyGraph.svg?label=Python
   :target: https://pypi.org/project/Python-EasyGraph/

.. image:: https://img.shields.io/pypi/l/Python-EasyGraph?label=License
   :target: https://github.com/easy-graph/Easy-Graph/blob/master/LICENSE

.. image:: https://static.pepy.tech/personalized-badge/python-easygraph?period=total&units=international_system&left_color=brightgreen&right_color=yellowgreen&left_text=Downloads
   :target: https://pypi.org/project/Python-EasyGraph/

- **Documentation:** https://easy-graph.github.io/
- **Source Code:** https://github.com/easy-graph/Easy-Graph
- **Issue Tracker:** https://github.com/easy-graph/Easy-Graph/issues
- **PyPI Homepage:** https://pypi.org/project/Python-EasyGraph/
- **Youtube channel:** https://www.youtube.com/@python-easygraph

Introduction
------------
**EasyGraph** is an open-source network analysis library. It is mainly written in Python and supports analysis for undirected networks and directed networks. EasyGraph supports various formats of network data and covers a series of important network analysis algorithms for community detection, structural hole spanner detection, network embedding, and motif detection. Moreover, EasyGraph implements some key elements using C++ and introduces multiprocessing optimization to achieve better efficiency.

New Features in Version 1.1
------------
- **Support for more hypergraph metrics and algorithms.** Such as `hypercoreness <https://www.nature.com/articles/s41467-023-41887-2>`_, `vector-centrality <https://www.sciencedirect.com/science/article/pii/S0960077922006075>`_, `s-centrality <https://epjds.epj.org/articles/epjdata/abs/2020/01/13688_2020_Article_231/13688_2020_Article_231.html>`_, and so on.
- **Support for more hypergraph datasets.** `Static hypergraph datasets and dynamic datasets <https://easy-graph.github.io/docs/reference/easygraph.datasets.html>`_ can be both loaded by calling corresponding dataset name.
- **Support for more flexible dynamic hypergraph visualization.** Users can define dynamic hypergraphs and visualize the structure of the hypergraph at each timestamp.
- **Support for more efficient hypergraph computation and hypergraph learning** Adoption of suitable storage structure and caching strategy for different metrics/hypergraph neural networks.

If you need more details, please see our `documentation <https://easy-graph.github.io/>`_ of latest version.

News
----
- [02-05-2024] We release EasyGraph 1.1! This version features hypergraph analysis and learning for higher-order network modeling and representation.
- [08-17-2023] We release EasyGraph 1.0!
- [08-08-2023] Our paper "EasyGraph: A Multifunctional, Cross-Platform, and Effective Library for Interdisciplinary Network Analysis" has been accepted by Patterns!

Stargazers
----------
.. image:: https://reporoster.com/stars/easy-graph/Easy-Graph
   :target: https://github.com/easy-graph/Easy-Graph/stargazers
   :alt: Stargazers repo roster for @easy-graph/Easy-Graph

Install
-------

.. The current version on PyPI is outdated, we'll push the latest version as soon as we figure out how to integrate the C++ binding framework we use with our CI pipeline.

.. In the meantime, here's a work around you can try to install the latest version of easygraph on your machine:

- **Prerequisites**

``3.8 <= Python <= 3.11`` is required.

.. Installation with ``pip`` (outdated)

- **Installation with** ``pip``

.. code::

    $ pip install --upgrade Python-EasyGraph

The conda package is no longer updated or maintained.

If you've installed EasyGraph this way before, please uninstall it with ``conda`` and install it with ``pip``.

If prebuilt EasyGraph wheels are not supported for your platform (OS / CPU arch, check `here <https://pypi.org/simple/python-easygraph/>`_), you can build it locally this way:

.. code:: bash

    git clone https://github.com/easy-graph/Easy-Graph && cd Easy-Graph && git checkout pybind11
    pip install pybind11
    python3 setup.py build_ext
    python3 setup.py install

- **Hint**

    EasyGraph uses  1.12.1 <= `PyTorch <https://pytorch.org/get-started/locally/>`_ < 2.0 for machine
    learning functions.
    Note that this does not prevent your from running non-machine learning functions normally,
    if there is no PyTorch in your environment.
    But you will receive some warnings which remind you some unavailable modules when they depend on it.

Simple Example
--------------


This example shows the general usage of methods in EasyGraph.

.. code:: python

  >>> import easygraph as eg
  >>> G = eg.Graph()
  >>> G.add_edges([(1,2), (2,3), (1,3), (3,4), (4,5), (3,5), (5,6)])
  >>> eg.pagerank(G)
  {1: 0.14272233049003707, 2: 0.14272233049003694, 3: 0.2685427766200994, 4: 0.14336430577918527, 5: 0.21634929087322705, 6: 0.0862989657474143}

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

Citation
--------

If you use EasyGraph in a scientific publication, we would appreciate citations to the following paper:

.. code:: bash

  @article{gao2023easygraph,
      title={{EasyGraph: A Multifunctional, Cross-Platform, and Effective Library for Interdisciplinary Network Analysis}},
      author={Min Gao and Zheng Li and Ruichen Li and Chenhao Cui and Xinyuan Chen and Bodian Ye and Yupeng Li and Weiwei Gu and Qingyuan Gong and Xin Wang and Yang Chen},
      year={2023},
      journal={Patterns},
      volume={4},
      number={10}
  }
