import random

from copy import deepcopy
from typing import List
from typing import Optional
from typing import Union

import easygraph as eg


__all__ = [
    "draw_SHS_center",
    "draw_SHS_center_kk",
    "draw_kamada_kawai",
    "draw_hypergraph",
]

from easygraph.functions.drawing.defaults import default_hypergraph_strength
from easygraph.functions.drawing.defaults import default_hypergraph_style
from easygraph.functions.drawing.defaults import default_size
from easygraph.functions.drawing.layout import force_layout
from easygraph.functions.drawing.utils import draw_circle_edge
from easygraph.functions.drawing.utils import draw_vertex


def draw_hypergraph(
    hg: "eg.Hypergraph",
    e_style: str = "circle",
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 1.0,
    v_color: Union[str, list] = "r",
    v_line_width: Union[str, list] = 1.0,
    e_color: Union[str, list] = "gray",
    e_fill_color: Union[str, list] = "whitesmoke",
    e_line_width: Union[str, list] = 1.0,
    font_size: float = 1.0,
    font_family: str = "sans-serif",
    push_v_strength: float = 1.0,
    push_e_strength: float = 1.0,
    pull_e_strength: float = 1.0,
    pull_center_strength: float = 1.0,
):
    r"""Draw the hypergraph structure.

    Args:
        ``hg`` (``eg.Hypergraph``): The EasyGraph's hypergraph object.
        ``e_style`` (``str``): The style of hyperedges. The available styles are only ``'circle'``. Defaults to ``'circle'``.
        ``v_label`` (``list``): The labels of vertices. Defaults to ``None``.
        ``v_size`` (``float`` or ``list``): The size of vertices. Defaults to ``1.0``.
        ``v_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices. Defaults to ``'r'``.
        ``v_line_width`` (``float`` or ``list``): The line width of vertices. Defaults to ``1.0``.
        ``e_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'gray'``.
        ``e_fill_color`` (``str`` or ``list``): The fill `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'whitesmoke'``.
        ``e_line_width`` (``float`` or ``list``): The line width of hyperedges. Defaults to ``1.0``.
        ``font_size`` (``float``): The font size of labels. Defaults to ``1.0``.
        ``font_family`` (``str``): The font family of labels. Defaults to ``'sans-serif'``.
        ``push_v_strength`` (``float``): The strength of pushing vertices. Defaults to ``1.0``.
        ``push_e_strength`` (``float``): The strength of pushing hyperedges. Defaults to ``1.0``.
        ``pull_e_strength`` (``float``): The strength of pulling hyperedges. Defaults to ``1.0``.
        ``pull_center_strength`` (``float``): The strength of pulling vertices to the center. Defaults to ``1.0``.
    """
    import matplotlib.pyplot as plt

    assert isinstance(
        hg, eg.Hypergraph
    ), "The input object must be a DHG's hypergraph object."
    assert e_style in ["circle"], "e_style must be 'circle'"
    assert hg.num_e > 0, "g must be a non-empty structure"
    fig, ax = plt.subplots(figsize=(6, 6))

    num_v, e_list = hg.num_v, deepcopy(hg.e[0])
    # default configures
    v_color, e_color, e_fill_color = default_hypergraph_style(
        hg.num_v, hg.num_e, v_color, e_color, e_fill_color
    )
    v_size, v_line_width, e_line_width, font_size = default_size(
        num_v, e_list, v_size, v_line_width, e_line_width
    )
    (
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_center_strength,
    ) = default_hypergraph_strength(
        num_v,
        e_list,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_center_strength,
    )
    # layout
    v_coor = force_layout(
        num_v,
        e_list,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_center_strength,
    )
    if e_style == "circle":
        draw_circle_edge(
            ax,
            v_coor,
            v_size,
            e_list,
            e_color,
            e_fill_color,
            e_line_width,
        )
    else:
        raise ValueError("e_style must be 'circle'")

    draw_vertex(
        ax,
        v_coor,
        v_label,
        font_size,
        font_family,
        v_size,
        v_color,
        v_line_width,
    )

    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")
    fig.tight_layout()
    plt.show()


def draw_SHS_center(G, SHS, rate=1, style="side"):
    """
    Draw the graph whose the SH Spanners are in the center, with random layout.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    rate : float
       The proportion of visible points and edges to the total

    style : string
        "side"- the label is next to the dot
        "center"- the label is in the center of the dot

    Returns
    -------
    graph : network
        the graph whose the SH Spanners are in the center.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pos = eg.random_position(G)
    center = np.zeros((len(SHS), 2), float)
    node = np.zeros((len(pos) - len(SHS), 2), float)
    m, n = 0, 0
    if rate == 1:
        for i in pos:
            if i in SHS:
                center[n][0] = 0.5 + (-1) ** random.randint(1, 2) * pos[i][0] / 5
                center[n][1] = 0.5 + (-1) ** random.randint(1, 2) * pos[i][1] / 5
                pos[i][0] = center[n][0]
                pos[i][1] = center[n][1]
                n += 1
            else:
                node[m][0] = pos[i][0]
                node[m][1] = pos[i][1]
                m += 1
        if style == "side":
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=10)
            plt.scatter(center[:, 0], center[:, 1], marker="*", color="r", s=20)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="None",
                edgecolors="b",
                s=50,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="o",
                color="None",
                edgecolors="r",
                s=50,
                linewidth=0.5,
            )
        k = 0
        for i in pos:
            if style == "side":
                plt.text(
                    pos[i][0],
                    pos[i][1],
                    i,
                    fontsize=5,
                    verticalalignment="top",
                    horizontalalignment="right",
                )
            elif style == "center":
                plt.text(
                    pos[i][0],
                    pos[i][1],
                    i,
                    fontsize=5,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
            k += 1
        for i in G.edges:
            p1 = [pos[i[0]][0], pos[i[1]][0]]
            p2 = [pos[i[0]][1], pos[i[1]][1]]
            plt.plot(p1, p2, "k-", alpha=0.3, linewidth=0.5)
        plt.show()

    else:
        degree = G.degree()
        sorted_degree = sorted(degree.items(), key=lambda d: d[1], reverse=True)
        l = int(rate * len(G))
        s = []
        for i in sorted_degree:
            if len(s) < l:
                s.append(i[0])
        for i in pos:
            if i in SHS and i in s:
                center[n][0] = 0.5 + (-1) ** random.randint(1, 2) * pos[i][0] / 5
                center[n][1] = 0.5 + (-1) ** random.randint(1, 2) * pos[i][1] / 5
                pos[i][0] = center[n][0]
                pos[i][1] = center[n][1]
                n += 1
            elif i in s:
                node[m][0] = pos[i][0]
                node[m][1] = pos[i][1]
                m += 1
        node = node[0:m, :]
        center = center[0:n, :]
        if style == "side":
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=10)
            plt.scatter(center[:, 0], center[:, 1], marker="*", color="r", s=20)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="None",
                edgecolors="b",
                s=50,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="o",
                color="None",
                edgecolors="r",
                s=50,
                linewidth=0.5,
            )
        k = 0
        for i in pos:
            if i in s:
                if style == "side":
                    plt.text(
                        pos[i][0],
                        pos[i][1],
                        i,
                        fontsize=5,
                        verticalalignment="top",
                        horizontalalignment="right",
                    )
                elif style == "center":
                    plt.text(
                        pos[i][0],
                        pos[i][1],
                        i,
                        fontsize=5,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                k += 1
        for i in G.edges:
            (u, v, t) = i
            if u in s and v in s:
                p1 = [pos[i[0]][0], pos[i[1]][0]]
                p2 = [pos[i[0]][1], pos[i[1]][1]]
                plt.plot(p1, p2, "k-", alpha=0.3, linewidth=0.5)
        plt.show()
    return


def draw_SHS_center_kk(G, SHS, rate=1, style="side"):
    """
    Draw the graph whose the SH Spanners are in the center, with a Kamada-Kawai force-directed layout.

    Parameters
    ----------
    G : graph
        A easygraph graph.

    SHS : list
        The SH Spanners in graph G.

    rate : float
       The proportion of visible points and edges to the total

    style : string
        "side"- the label is next to the dot
        "center"- the label is in the center of the dot

    Returns
    -------
    graph : network
        the graph whose the SH Spanners are in the center.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pos = eg.kamada_kawai_layout(G)
    center = np.zeros((len(SHS), 2), float)
    node = np.zeros((len(pos) - len(SHS), 2), float)
    m, n = 0, 0
    if rate == 1:
        for i in pos:
            if i in SHS:
                center[n][0] = pos[i][0] / 5
                center[n][1] = pos[i][1] / 5
                pos[i][0] = center[n][0]
                pos[i][1] = center[n][1]
                n += 1
            else:
                node[m][0] = pos[i][0]
                node[m][1] = pos[i][1]
                m += 1
        if style == "side":
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=10)
            plt.scatter(center[:, 0], center[:, 1], marker="*", color="r", s=20)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="None",
                edgecolors="b",
                s=50,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="o",
                color="None",
                edgecolors="r",
                s=50,
                linewidth=0.5,
            )
        k = 0
        for i in pos:
            if style == "side":
                plt.text(
                    pos[i][0],
                    pos[i][1],
                    i,
                    fontsize=5,
                    verticalalignment="top",
                    horizontalalignment="right",
                )
            elif style == "center":
                plt.text(
                    pos[i][0],
                    pos[i][1],
                    i,
                    fontsize=5,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
            k += 1
        for i in G.edges:
            p1 = [pos[i[0]][0], pos[i[1]][0]]
            p2 = [pos[i[0]][1], pos[i[1]][1]]
            plt.plot(p1, p2, "k-", alpha=0.3, linewidth=0.5)
        plt.show()
    else:
        degree = G.degree()
        sorted_degree = sorted(degree.items(), key=lambda d: d[1], reverse=True)
        l = int(rate * len(G))
        s = []
        for i in sorted_degree:
            if len(s) < l:
                s.append(i[0])
        for i in pos:
            if i in SHS and i in s:
                center[n][0] = pos[i][0] / 5
                center[n][1] = pos[i][1] / 5
                pos[i][0] = center[n][0]
                pos[i][1] = center[n][1]
                n += 1
            elif i in s:
                node[m][0] = pos[i][0]
                node[m][1] = pos[i][1]
                m += 1
        node = node[0:m, :]
        center = center[0:n, :]
        if style == "side":
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=10)
            plt.scatter(center[:, 0], center[:, 1], marker="*", color="r", s=20)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="None",
                edgecolors="b",
                s=50,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="o",
                color="None",
                edgecolors="r",
                s=50,
                linewidth=0.5,
            )
        k = 0
        for i in pos:
            if i in s:
                if style == "side":
                    plt.text(
                        pos[i][0],
                        pos[i][1],
                        i,
                        fontsize=5,
                        verticalalignment="top",
                        horizontalalignment="right",
                    )
                elif style == "center":
                    plt.text(
                        pos[i][0],
                        pos[i][1],
                        i,
                        fontsize=5,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                k += 1
        for i in G.edges:
            (u, v, t) = i
            if u in s and v in s:
                p1 = [pos[i[0]][0], pos[i[1]][0]]
                p2 = [pos[i[0]][1], pos[i[1]][1]]
                plt.plot(p1, p2, "k-", alpha=0.3, linewidth=0.5)
        plt.show()
    return


def draw_kamada_kawai(G, rate=1, style="side"):
    """Draw the graph G with a Kamada-Kawai force-directed layout.

    Parameters
    ----------
    G : graph
       A easygraph graph

    rate : float
       The proportion of visible points and edges to the total

    style : string
        "side"- the label is next to the dot
        "center"- the label is in the center of the dot

    """
    import matplotlib.pyplot as plt
    import numpy as np

    pos = eg.kamada_kawai_layout(G)
    node = np.zeros((len(pos), 2), float)
    m, n = 0, 0
    if rate == 1:
        for i in pos:
            node[m][0] = pos[i][0]
            node[m][1] = pos[i][1]
            m += 1
        if style == "side":
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=10)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="None",
                edgecolors="b",
                s=50,
                linewidth=0.5,
            )
        k = 0
        for i in pos:
            if style == "side":
                plt.text(
                    pos[i][0],
                    pos[i][1],
                    i,
                    fontsize=5,
                    verticalalignment="top",
                    horizontalalignment="right",
                )
            elif style == "center":
                plt.text(
                    pos[i][0],
                    pos[i][1],
                    i,
                    fontsize=5,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
            k += 1
        for i in G.edges:
            p1 = [pos[i[0]][0], pos[i[1]][0]]
            p2 = [pos[i[0]][1], pos[i[1]][1]]
            plt.plot(p1, p2, "k-", alpha=0.3, linewidth=0.5)
        plt.show()
    else:
        degree = G.degree()
        sorted_degree = sorted(degree.items(), key=lambda d: d[1], reverse=True)
        l = int(rate * len(G))
        s = []
        for i in sorted_degree:
            if len(s) < l:
                s.append(i[0])
        for i in pos:
            if i in s:
                node[m][0] = pos[i][0]
                node[m][1] = pos[i][1]
                m += 1
        node = node[0:m, :]
        if style == "side":
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=10)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="None",
                edgecolors="b",
                s=50,
                linewidth=0.5,
            )
        k = 0
        for i in pos:
            if i in s:
                if style == "side":
                    plt.text(
                        pos[i][0],
                        pos[i][1],
                        i,
                        fontsize=5,
                        verticalalignment="top",
                        horizontalalignment="right",
                    )
                elif style == "center":
                    plt.text(
                        pos[i][0],
                        pos[i][1],
                        i,
                        fontsize=5,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                k += 1
        for i in G.edges:
            (u, v, t) = i
            if u in s and v in s:
                p1 = [pos[i[0]][0], pos[i[1]][0]]
                p2 = [pos[i[0]][1], pos[i[1]][1]]
                plt.plot(p1, p2, "k-", alpha=0.3, linewidth=0.5)
        plt.show()
    return


if __name__ == "__main__":
    G = eg.datasets.get_graph_karateclub()
    draw_SHS_center(G, [1, 33, 34], style="side")
    draw_SHS_center(G, [1, 33, 34], style="center")
    draw_SHS_center_kk(G, [1, 33, 34], style="side")
    draw_SHS_center_kk(G, [1, 33, 34], style="center")
    draw_kamada_kawai(G, style="side")
    draw_kamada_kawai(G, style="center")
    draw_SHS_center(G, [1, 33, 34], rate=0.8, style="side")
    draw_SHS_center(G, [1, 33, 34], rate=0.8, style="center")
    draw_SHS_center_kk(G, [1, 33, 34], rate=0.8, style="side")
    draw_SHS_center_kk(G, [1, 33, 34], rate=0.8, style="center")
    draw_kamada_kawai(G, rate=0.8, style="side")
    draw_kamada_kawai(G, rate=0.8, style="center")
