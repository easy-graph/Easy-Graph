# This file is part of the NetworkX distribution.

# NetworkX is distributed with the 3-clause BSD license.


# ::
#    Copyright (C) 2004-2022, NetworkX Developers
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.

#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:

#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#      * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#      * Neither the name of the NetworkX Developers nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
*****
Pajek
*****
Read graphs in Pajek format.

This implementation handles directed and undirected graphs including
those with self loops and parallel edges.

Format
------
See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
for format information.

"""

import warnings

import easygraph as eg

# import networkx as nx
from easygraph.utils import open_file


__all__ = ["read_pajek", "parse_pajek", "generate_pajek", "write_pajek"]


def generate_pajek(G):
    """Generate lines in Pajek graph format.

    Parameters
    ----------
    G : graph
       A EasyGraph graph

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    if G.name == "":
        name = "EasyGraph"
    else:
        name = G.name
    # Apparently many Pajek format readers can't process this line
    # So we'll leave it out for now.
    # yield '*network %s'%name

    # write nodes with attributes
    yield f"*vertices {G.order()}"
    nodes = list(G)
    # make dictionary mapping nodes to integers
    nodenumber = dict(zip(nodes, range(1, len(nodes) + 1)))
    for n in nodes:
        # copy node attributes and pop mandatory attributes
        # to avoid duplication.
        na = G.nodes.get(n, {}).copy()
        x = na.pop("x", 0.0)
        y = na.pop("y", 0.0)
        try:
            id = int(na.pop("id", nodenumber[n]))
        except ValueError as err:
            err.args += (
                "Pajek format requires 'id' to be an int()."
                " Refer to the 'Relabeling nodes' section.",
            )
            raise
        nodenumber[n] = id
        shape = na.pop("shape", "ellipse")
        s = " ".join(map(make_qstr, (id, n, x, y, shape)))
        # only optional attributes are left in na.
        for k, v in na.items():
            if isinstance(v, str) and v.strip() != "":
                s += f" {make_qstr(k)} {make_qstr(v)}"
            else:
                warnings.warn(
                    f"Node attribute {k} is not processed."
                    f" {('Empty attribute' if isinstance(v, str) else 'Non-string attribute')}."
                )
        yield s

    # write edges with attributes
    if G.is_directed():
        yield "*arcs"
    else:
        yield "*edges"
    # from icecream import ic
    # ic(G.edges)
    # if isinstance(G, MultiGraph)
    for u, v, *edgedata in G.edges:
        # if len(edgedata) > 1:
        #     edgedata = edgedata[1]
        # else:
        #     edgedata = edgedata[0]
        edgedata = edgedata[-1]
        d = edgedata.copy()
        value = d.pop("weight", 1.0)  # use 1 as default edge value
        s = " ".join(map(make_qstr, (nodenumber[u], nodenumber[v], value)))
        for k, v in d.items():
            if isinstance(v, str) and v.strip() != "":
                s += f" {make_qstr(k)} {make_qstr(v)}"
            else:
                warnings.warn(
                    f"Edge attribute {k} is not processed."
                    f" {('Empty attribute' if isinstance(v, str) else 'Non-string attribute')}."
                )
        yield s


@open_file(1, mode="wb")
def write_pajek(G, path, encoding="UTF-8"):
    """Write graph in Pajek format to path.

    Parameters
    ----------
    G : graph
       A EasyGraph graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.

    Examples
    --------
    >>> G = eg.path_graph(4)
    >>> eg.write_pajek(G, "test.net")

    Warnings
    --------
    Optional node attributes and edge attributes must be non-empty strings.
    Otherwise it will not be written into the file. You will need to
    convert those attributes to strings if you want to keep them.

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    for line in generate_pajek(G):
        line += "\n"
        path.write(line.encode(encoding))


@open_file(0, mode="rb")
def read_pajek(path):
    """Read graph in Pajek format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be uncompressed.

    Returns
    -------
    G : EasyGraph MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> G = eg.path_graph(4)
    >>> eg.write_pajek(G, "test.net")
    >>> G = eg.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1 = eg.Graph(G)

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    lines = (line.decode() for line in path)
    # with open(path) as f:
    #     lines = f.readlines()
    return parse_pajek(lines)


def parse_pajek(lines):
    """Parse Pajek format graph from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in Pajek format.

    Returns
    -------
    G : EasyGraph graph

    See Also
    --------
    read_pajek

    """
    import shlex

    # multigraph=False
    if isinstance(lines, str):
        lines = iter(lines.split("\n"))
    # from itertools import tee
    # lines, lines2 = tee(lines)
    # from icecream import ic
    # ic(next(lines2))
    lines = iter([line.rstrip("\n") for line in lines])
    G = eg.MultiDiGraph()  # are multiedges allowed in Pajek? assume yes
    labels = []  # in the order of the file, needed for matrix
    while lines:
        try:
            l = next(lines)
        except:  # EOF
            break
        if l.lower().startswith("*network"):
            try:
                label, name = l.split(None, 1)
            except ValueError:
                # Line was not of the form:  *network NAME
                pass
            else:
                G.graph["name"] = name
        elif l.lower().startswith("*vertices"):
            nodelabels = {}
            l, nnodes = l.split()
            for i in range(int(nnodes)):
                l = next(lines)
                try:
                    splitline = [x for x in shlex.split(str(l))]
                except AttributeError:
                    splitline = shlex.split(str(l))
                id, label = splitline[0:2]
                labels.append(label)
                G.add_node(label)
                nodelabels[id] = label
                G.nodes[label]["id"] = id
                try:
                    x, y, shape = splitline[2:5]
                    G.nodes[label].update(
                        {"x": float(x), "y": float(y), "shape": shape}
                    )
                except:
                    pass
                extra_attr = zip(splitline[5::2], splitline[6::2])
                G.nodes[label].update(extra_attr)
        elif l.lower().startswith("*edges") or l.lower().startswith("*arcs"):
            if l.lower().startswith("*edge"):
                # switch from multidigraph to multigraph
                G = eg.MultiGraph(G)
            if l.lower().startswith("*arcs"):
                # switch to directed with multiple arcs for each existing edge
                # G = G.to_directed()
                pass
            for l in lines:
                try:
                    splitline = [x for x in shlex.split(str(l))]
                except AttributeError:
                    splitline = shlex.split(str(l))

                if len(splitline) < 2:
                    continue
                ui, vi = splitline[0:2]
                u = nodelabels.get(ui, ui)
                v = nodelabels.get(vi, vi)
                # parse the data attached to this edge and put in a dictionary
                edge_data = {}
                try:
                    # there should always be a single value on the edge?
                    w = splitline[2:3]
                    edge_data.update({"weight": float(w[0])})
                except:
                    pass
                    # if there isn't, just assign a 1
                #                    edge_data.update({'value':1})
                extra_attr = zip(splitline[3::2], splitline[4::2])
                edge_data.update(extra_attr)
                # if G.has_edge(u,v):
                #     multigraph=True
                G.add_edge(u, v, **edge_data)
        elif l.lower().startswith("*matrix"):
            G = eg.DiGraph(G)
            adj_list = (
                (labels[row], labels[col], {"weight": int(data)})
                for (row, line) in enumerate(lines)
                for (col, data) in enumerate(line.split())
                if int(data) != 0
            )
            G.add_edges_from(adj_list)

    return G


def make_qstr(t):
    """Returns the string representation of t.
    Add outer double-quotes if the string has a space.
    """
    if not isinstance(t, str):
        t = str(t)
    if " " in t:
        t = f'"{t}"'
    return t
