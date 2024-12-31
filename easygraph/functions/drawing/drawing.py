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
    "draw_dynamic_hypergraph",
    "draw_easygraph_nodes",
    "draw_easygraph_edges",
    "draw_louvain_com",
    "draw_lpa_com",
    "draw_gm_com",
    "draw_ego_graph",
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
    ), "The input object must be a EasyGraph's hypergraph object."
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
        e_color,
        v_line_width,
    )

    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")
    fig.tight_layout()
    plt.show()


def _draw_single_dynamic_hypergraph(
    hg: "eg.Hypergraph",
    ax,
    title_font_size=4,
    group_name: str = "main",
    e_style: str = "circle",
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 2.0,
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
    import matplotlib.pyplot as plt

    assert isinstance(
        hg, eg.Hypergraph
    ), "The input object must be a EasyGraph's hypergraph object."
    assert e_style in ["circle"], "e_style must be 'circle'"
    assert hg.num_e > 0, "g must be a non-empty structure"

    num_v, e_list = hg.num_v, deepcopy(hg.e_of_group(group_name)[0])
    # default configures
    v_color, e_color, e_fill_color = default_hypergraph_style(
        hg.num_v, hg.num_e, v_color, e_color, e_fill_color
    )
    v_size, v_line_width, e_line_width, font_size = default_size(
        num_v, e_list, v_size, v_line_width, e_line_width, font_size
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
        v_color,
        v_line_width,
    )
    plt.title(group_name, fontsize=title_font_size)
    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")


def draw_dynamic_hypergraph(
    G,
    group_name_list=None,
    column_size=None,
    save_path=None,
    title_font_size=4,
    e_style: str = "circle",
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 2.0,
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
    """

    Parameters
    ----------
    G eg.Hypergraph
    group_name_list The groups to visualize
    column_size The number of subplots placed in each row
    save_path path to save visualization
    title_font_size The font size of tilte of each subplot

    """
    import math

    import matplotlib.pyplot as plt

    # if group_name_list == None:
    #     group_name_list = G.group_names
    COLUMN_SIZE = 3 if column_size == None else column_size
    ROW_SIZE = math.ceil(len(group_name_list) / COLUMN_SIZE)
    fig = plt.figure()

    sub = 1
    for gn in group_name_list:
        if sub > len(group_name_list):
            break
        tmp_ax = fig.add_subplot(ROW_SIZE, COLUMN_SIZE, sub)
        _draw_single_dynamic_hypergraph(
            G,
            ax=tmp_ax,
            group_name=gn,
            title_font_size=title_font_size,
            e_style=e_style,
            v_label=v_label,
            v_size=v_size,
            v_color=v_color,
            v_line_width=v_line_width,
            e_color=e_color,
            e_fill_color=e_fill_color,
            e_line_width=e_line_width,
            font_size=font_size,
            font_family=font_family,
            push_v_strength=push_v_strength,
            push_e_strength=push_e_strength,
            pull_e_strength=pull_e_strength,
            pull_center_strength=pull_center_strength,
        )
        sub += 1
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def draw_easygraph_nodes(
    G,
    pos,
    nodelist=None,
    node_size=300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
    margins=None,
):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
        A EasyGraph graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default list(G))
        Draw only specified nodes

    node_size : scalar or array (default=300)
        Size of nodes.  If an array it must be the same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or array of floats (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap (default=None)
        Colormap for mapping intensities of nodes

    vmin,vmax : floats or None (default=None)
        Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence] (default=1.0)
        Line width of symbol border

    edgecolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders. Can be a single color or a sequence of colors with the
        same length as nodelist. Color can be string or rgb (or rgba) tuple of floats
        from 0-1. If numeric values are specified they will be mapped to colors
        using the cmap and vmin,vmax parameters. See `~matplotlib.pyplot.scatter` for more details.

    label : [None | string]
        Label for legend

    margins : float or 2-tuple, optional
        Sets the padding for axis autoscaling. Increase margin to prevent
        clipping for nodes that are near the edges of an image. Values should
        be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`
        for details. The default is `None`, which uses the Matplotlib default.

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G = eg.dodecahedral_graph()
    >>> nodes = eg.draw_easygraph_nodes(G, pos=eg.spring_layout(G))



    """
    from collections.abc import Iterable

    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if len(nodelist) == 0:  # empty nodelist, no drawing
        return mpl.collections.PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as err:
        raise eg.EasygraphError(f"Node {err} has no position.") from err

    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
        alpha = None

    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    if margins is not None:
        if isinstance(margins, Iterable):
            ax.margins(*margins)
        else:
            ax.margins(margins)

    node_collection.set_zorder(2)
    return node_collection


def draw_easygraph_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=None,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle="arc3",
    min_source_margin=0,
    min_target_margin=0,
):
    r"""Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
        A easygraph graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edgelist : collection of edge tuples (default=G.edges())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string or array of strings (default='solid')
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        Can be a single style or a sequence of styles with the same
        length as the edge list.
        If less styles than edges are given the styles will cycle.
        If more styles than edges are given the styles will be used sequentially
        and not be exhausted.
        Also, `(offset, onoffseq)` tuples can be used as style instead of a strings.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or array of floats (default=None)
        The edge transparency.  This can be a single alpha value,
        in which case it will be applied to all specified edges. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).

        Note: Arrowheads will be the same color as edges.

    arrowstyle : str (default='-\|>' for directed graphs)
        For directed graphs and `arrows==True` defaults to '-\|>',
        For undirected graphs default to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    connectionstyle : string (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of 'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the beginning of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
     matplotlib.collections.LineCollection or a list of matplotlib.patches.FancyArrowPatch
        If ``arrows=True``, a list of FancyArrowPatches is returned.
        If ``arrows=False``, a LineCollection is returned.
        If ``arrows=None`` (the default), then a LineCollection is returned if
        `G` is undirected, otherwise returns a list of FancyArrowPatches.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.

    Self-loops are always drawn with `~matplotlib.patches.FancyArrowPatch`
    regardless of the value of `arrows` or whether `G` is directed.
    When ``arrows=False`` or ``arrows=None`` and `G` is undirected, the
    FancyArrowPatches corresponding to the self-loops are not explicitly
    returned. They should instead be accessed via the ``Axes.patches``
    attribute (see examples).

    """
    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.colors  # call as mpl.colors
    import matplotlib.patches  # call as mpl.patches
    import matplotlib.path  # call as mpl.path
    import matplotlib.pyplot as plt
    import numpy as np

    # The default behavior is to use LineCollection to draw edges for
    # undirected graphs (for performance reasons) and use FancyArrowPatches
    # for directed graphs.
    # The `arrows` keyword can be used to override the default behavior
    use_linecollection = not G.is_directed()
    if arrows in (True, False):
        use_linecollection = not arrows

    # Some kwargs only apply to FancyArrowPatches. Warn users when they use
    # non-default values for these kwargs when LineCollection is being used
    # instead of silently ignoring the specified option
    if use_linecollection and any(
        [
            arrowstyle is not None,
            arrowsize != 10,
            connectionstyle != "arc3",
            min_source_margin != 0,
            min_target_margin != 0,
        ]
    ):
        import warnings

        msg = (
            "\n\nThe {0} keyword argument is not applicable when drawing edges\n"
            "with LineCollection.\n\n"
            "To make this warning go away, either specify `arrows=True` to\n"
            "force FancyArrowPatches or use the default value for {0}.\n"
            "Note that using FancyArrowPatches may be slow for large graphs.\n"
        )
        if arrowstyle is not None:
            msg = msg.format("arrowstyle")
        if arrowsize != 10:
            msg = msg.format("arrowsize")
        if connectionstyle != "arc3":
            msg = msg.format("connectionstyle")
        if min_source_margin != 0:
            msg = msg.format("min_source_margin")
        if min_target_margin != 0:
            msg = msg.format("min_target_margin")
        warnings.warn(msg, category=UserWarning, stacklevel=2)

    if arrowstyle == None:
        if G.is_directed():
            arrowstyle = "-|>"
        else:
            arrowstyle = "-"

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges)

    if len(edgelist) == 0:  # no edges!
        return []

    if nodelist is None:
        nodelist = list(G.nodes)

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"
    edgelist_tuple = list(map(tuple, edgelist))

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.all([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    def _draw_networkx_edges_line_collection():
        edge_collection = mpl.collections.LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            alpha=alpha,
        )
        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)
        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    def _draw_networkx_edges_fancy_arrow_patch():
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []

        if isinstance(arrowsize, list):
            if len(arrowsize) != len(edge_pos):
                raise ValueError("arrowsize should have the same length as edgelist")
        else:
            mutation_scale = arrowsize  # scale factor of arrow head

        base_connection_style = mpl.patches.ConnectionStyle(connectionstyle)

        # Fallback for self-loop scale. Left outside of _connectionstyle so it is
        # only computed once
        max_nodesize = np.array(node_size).max()

        def _connectionstyle(posA, posB, *args, **kwargs):
            # check if we need to do a self-loop
            if np.all(posA == posB):
                # Self-loops are scaled by view extent, except in cases the extent
                # is 0, e.g. for a single node. In this case, fall back to scaling
                # by the maximum node size
                selfloop_ht = 0.005 * max_nodesize if h == 0 else h
                # this is called with _screen space_ values so convert back
                # to data space
                data_loc = ax.transData.inverted().transform(posA)
                v_shift = 0.1 * selfloop_ht
                h_shift = v_shift * 0.5
                # put the top of the loop first so arrow is not hidden by node
                path = [
                    # 1
                    data_loc + np.asarray([0, v_shift]),
                    # 4 4 4
                    data_loc + np.asarray([h_shift, v_shift]),
                    data_loc + np.asarray([h_shift, 0]),
                    data_loc,
                    # 4 4 4
                    data_loc + np.asarray([-h_shift, 0]),
                    data_loc + np.asarray([-h_shift, v_shift]),
                    data_loc + np.asarray([0, v_shift]),
                ]

                ret = mpl.path.Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])
            # if not, fall back to the user specified behavior
            else:
                ret = base_connection_style(posA, posB, *args, **kwargs)

            return ret

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in zip(fancy_edges_indices, edge_pos):
            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target

            if isinstance(arrowsize, list):
                # Scale each factor of each arrow based on arrowsize list
                mutation_scale = arrowsize[i]

            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) > i:
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) > i:
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            if (
                np.iterable(style)
                and not isinstance(style, str)
                and not isinstance(style, tuple)
            ):
                if len(style) > i:
                    linestyle = style[i]
                else:  # Cycle through styles
                    linestyle = style[i % len(style)]
            else:
                linestyle = style

            arrow = mpl.patches.FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=_connectionstyle,
                linestyle=linestyle,
                zorder=1,
            )  # arrows go behind nodes

            arrow_collection.append(arrow)
            ax.add_patch(arrow)

        return arrow_collection

    # compute initial view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - minx
    h = maxy - miny

    # Draw the edges
    if use_linecollection:
        edge_viz_obj = _draw_networkx_edges_line_collection()
        # Make sure selfloop edges are also drawn
        selfloops_to_draw = [loop for loop in eg.selfloop_edges(G) if loop in edgelist]
        if selfloops_to_draw:
            fancy_edges_indices = [
                edgelist_tuple.index(loop) for loop in selfloops_to_draw
            ]
            edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in selfloops_to_draw])
            arrowstyle = "-"
            _draw_networkx_edges_fancy_arrow_patch()
    else:
        fancy_edges_indices = range(len(edgelist))
        edge_viz_obj = _draw_networkx_edges_fancy_arrow_patch()

    # update view after drawing
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return edge_viz_obj


def draw_SHS_center(G, SHS, rate=1, style="center"):
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

    plt.figure(figsize=(8, 8))
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
                color="skyblue",
                edgecolors="skyblue",
                s=300,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="o",
                color="tomato",
                edgecolors="tomato",
                s=500,
                linewidth=0.5,
                zorder=2,
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
                    fontsize=10,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
            k += 1
        for i in G.edges:
            p1 = [pos[i[0]][0], pos[i[1]][0]]
            p2 = [pos[i[0]][1], pos[i[1]][1]]
            plt.plot(
                p1,
                p2,
                color="skyblue",
                linestyle="-",
                alpha=0.3,
                linewidth=1.8,
                zorder=1,
            )
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
                plt.plot(p1, p2, color="skyblue", linestyle="-", alpha=0.3, linewidth=3)
        plt.show()
    return


def draw_SHS_center_kk(G, SHS, rate=1, style="center"):
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
            plt.scatter(node[:, 0], node[:, 1], marker=".", color="b", s=50)
            plt.scatter(center[:, 0], center[:, 1], marker="*", color="r", s=100)
        elif style == "center":
            plt.scatter(
                node[:, 0],
                node[:, 1],
                marker="o",
                color="skyblue",
                edgecolors="skyblue",
                s=300,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="o",
                color="skyblue",
                edgecolors="skyblue",
                s=300,
                linewidth=0.5,
            )
            plt.scatter(
                center[:, 0],
                center[:, 1],
                marker="*",
                color="None",
                edgecolors="r",
                s=1000,
                linewidth=2,
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
                    fontsize=10,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
            k += 1
        for i in G.edges:
            p1 = [pos[i[0]][0], pos[i[1]][0]]
            p2 = [pos[i[0]][1], pos[i[1]][1]]
            plt.plot(p1, p2, color="skyblue", linestyle="-", alpha=0.3, linewidth=3)
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
                plt.plot(p1, p2, color="skyblue", linestyle="-", alpha=0.3, linewidth=3)
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


def draw_louvain_com(G, l_com):
    """
    Draw the graph and show the communities

    Parameters
    ----------
    G : graph
    l_com : communities created by louvain algorithm
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 8))
    n = len(l_com)
    colors = get_n_colors(n + 1)
    com_pos = community_pos(n)
    node = np.zeros((len(G.nodes), 2), float)
    node_idx = np.zeros(len(G.nodes) + 1)
    edge_label = edge_partition(G, l_com)
    k = 0

    for i in range(n):
        n_pos = node_pos(len(l_com[i]))
        com_list = list(l_com[i])
        m = len(com_list)
        start = k
        for j in range(m):
            node[k][0] = com_pos[i][0] + n_pos[j][0]
            node[k][1] = com_pos[i][1] + n_pos[j][1]
            node_idx[com_list[j]] = k
            k += 1
        plt.scatter(
            node[start:k, 0],
            node[start:k, 1],
            marker="o",
            color=colors[i],
            edgecolors=colors[i],
            s=300,
            linewidth=0.5,
            zorder=2,
        )
        for j in range(m):
            x = int(node_idx[com_list[j]])
            plt.text(
                node[x][0],
                node[x][1],
                com_list[j],
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="center",
                color="white",
            )
    for i in G.edges:
        x = int(node_idx[int(i[0])])
        y = int(node_idx[int(i[1])])
        p1 = [node[x][0], node[y][0]]
        p2 = [node[x][1], node[y][1]]
        plt.plot(
            p1,
            p2,
            color=colors[edge_label[(i[0], i[1])]],
            linestyle="-",
            alpha=0.3,
            linewidth=1.5,
            zorder=1,
        )
    plt.show()
    return


def draw_lpa_com(G, lpa_com):
    """
    Draw the graph and show the communities

    Parameters
    ----------
    G : graph
    lpa_com : communities created by LPA
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 8))
    list_lpa_com = list(lpa_com.values())
    n = len(list_lpa_com)
    colors = get_n_colors(n + 1)
    com_pos = community_pos(n)
    node = np.zeros((len(G.nodes), 2), float)
    node_idx = np.zeros(len(G.nodes) + 1)
    edge_label = edge_partition(G, list_lpa_com)
    k = 0

    for i in range(n):
        cur_com = list_lpa_com[i]
        m = len(cur_com)
        n_pos = node_pos(m)
        start = k
        for j in range(m):
            node[k][0] = com_pos[i][0] + n_pos[j][0]
            node[k][1] = com_pos[i][1] + n_pos[j][1]
            node_idx[cur_com[j]] = k
            k += 1
        plt.scatter(
            node[start:k, 0],
            node[start:k, 1],
            marker="o",
            color=colors[i],
            edgecolors=colors[i],
            s=300,
            linewidth=0.5,
            zorder=2,
        )
        for j in range(m):
            x = int(node_idx[cur_com[j]])
            plt.text(
                node[x][0],
                node[x][1],
                cur_com[j],
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="center",
                color="white",
            )
    for i in G.edges:
        x = int(node_idx[int(i[0])])
        y = int(node_idx[int(i[1])])
        p1 = [node[x][0], node[y][0]]
        p2 = [node[x][1], node[y][1]]
        plt.plot(
            p1,
            p2,
            color=colors[edge_label[(i[0], i[1])]],
            linestyle="-",
            alpha=0.3,
            linewidth=1.5,
            zorder=1,
        )
    plt.show()
    return


def draw_gm_com(G, gm_com):
    """
    Draw the graph and show the communities

    Parameters
    ----------
    G : graph
    gm_com : communities created by greedy modularity
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 8))
    list_gm_com = [list(i) for i in gm_com]
    n = len(list_gm_com)
    colors = get_n_colors(n + 1)
    com_pos = community_pos(n)
    node = np.zeros((len(G.nodes), 2), float)
    node_idx = np.zeros(len(G.nodes) + 1)
    edge_label = edge_partition(G, list_gm_com)
    k = 0

    for i in range(n):
        cur_com = list_gm_com[i]
        m = len(cur_com)
        n_pos = node_pos(m)
        start = k
        for j in range(m):
            node[k][0] = com_pos[i][0] + n_pos[j][0]
            node[k][1] = com_pos[i][1] + n_pos[j][1]
            node_idx[cur_com[j]] = k
            k += 1
        plt.scatter(
            node[start:k, 0],
            node[start:k, 1],
            marker="o",
            color=colors[i],
            edgecolors=colors[i],
            s=300,
            linewidth=0.5,
            zorder=2,
        )
        for j in range(m):
            x = int(node_idx[cur_com[j]])
            plt.text(
                node[x][0],
                node[x][1],
                cur_com[j],
                fontsize=10,
                verticalalignment="center",
                horizontalalignment="center",
                color="white",
            )
    for i in G.edges:
        x = int(node_idx[int(i[0])])
        y = int(node_idx[int(i[1])])
        p1 = [node[x][0], node[y][0]]
        p2 = [node[x][1], node[y][1]]
        plt.plot(
            p1,
            p2,
            color=colors[edge_label[(i[0], i[1])]],
            linestyle="-",
            alpha=0.3,
            linewidth=1.5,
            zorder=1,
        )
    plt.show()
    return


def get_n_colors(n):
    import numpy as np

    from matplotlib import cm

    viridis = cm.get_cmap("viridis", n)
    colors = viridis(np.linspace(0, 1, n))
    return colors


def community_pos(n, scale=10):
    """
    Set position for every community.

    Parameters
    ----------
    n : number of communities
    scale : parameter for sprint_layout
    """
    graph = eg.Graph()
    graph.add_nodes(range(n))
    pos = eg.spring_layout(graph, scale=scale)
    return pos


def node_pos(n, scale=2):
    """
    Set position for every node in a community

    Parameters
    ----------
    n : number of nodes
    scale : parameter for sprint_layout
    """
    graph = eg.Graph()
    graph.add_nodes(range(n))
    pos = eg.spring_layout(graph, scale=scale)
    return pos


def edge_partition(G, community):
    """
    Label every edge with the community it belongs to.

    Parameters
    ----------
    G : the graph
    community : communities of the graph
    """
    edge_label = {}
    n = len(community)
    for edge in G.edges:
        for i in range(n):
            if edge[0] in community[i] and edge[1] in community[i]:
                edge_label[(edge[0], edge[1])] = i
                break
            elif edge[0] in community[i] or edge[1] in community[i]:
                edge_label[(edge[0], edge[1])] = n
                break
    return edge_label


def draw_ego_graph(G, ego_graph):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 10))
    pos = eg.random_position(G)
    center = np.zeros((len(ego_graph), 2), float)
    node = np.zeros((len(pos) - len(ego_graph), 2), float)
    m, n = 0, 0
    for i in pos:
        if i in list(ego_graph.nodes.keys()):
            center[n][0] = 0.5 + (-1) ** np.random.randint(1, 2) * pos[i][0] / 3
            center[n][1] = 0.5 + (-1) ** np.random.randint(1, 2) * pos[i][1] / 3
            pos[i][0] = center[n][0]
            pos[i][1] = center[n][1]
            n += 1
        else:
            node[m][0] = pos[i][0]
            node[m][1] = pos[i][1]
            m += 1
    plt.scatter(
        node[:, 0],
        node[:, 1],
        marker="o",
        color="skyblue",
        edgecolors="skyblue",
        s=100,
        linewidth=0.5,
    )
    plt.scatter(
        center[:, 0],
        center[:, 1],
        marker="o",
        color="tomato",
        edgecolors="tomato",
        s=200,
        linewidth=0.5,
        zorder=2,
    )
    k = 0
    for i in pos:
        plt.text(
            pos[i][0],
            pos[i][1],
            i,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center",
        )
        k += 1
    for i in G.edges:
        p1 = [pos[i[0]][0], pos[i[1]][0]]
        p2 = [pos[i[0]][1], pos[i[1]][1]]
        if i not in ego_graph.edges:
            plt.plot(
                p1,
                p2,
                color="skyblue",
                linestyle="-",
                alpha=0.3,
                linewidth=1.8,
                zorder=1,
            )
        else:
            plt.plot(
                p1,
                p2,
                color="tomato",
                linestyle="-",
                alpha=0.3,
                linewidth=1.8,
                zorder=1,
            )
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
