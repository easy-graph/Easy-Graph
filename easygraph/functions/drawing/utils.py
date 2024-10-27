from itertools import chain
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial import ConvexHull

from .geometry import common_tangent_radian
from .geometry import polar_position
from .geometry import rad_2_deg
from .geometry import radian_from_atan
from .geometry import vlen


# from fa2 import ForceAtlas2
# import bezier
# import numpy as np
# from easygraph import to_networkx
# from easygraph.utils.exception import EasyGraphError
# import easygraph as eg


def safe_div(a: np.ndarray, b: np.ndarray, jitter_scale: float = 0.000001):
    mask = b == 0
    b[mask] = 1
    eps = 1e-10
    inv_b = np.divide(1.0, np.maximum(b, eps))
    res = a * inv_b
    if mask.sum() > 0:
        res[mask.repeat(2, 2)] = np.random.randn(mask.sum() * 2) * jitter_scale
    return res


def init_pos(num_v: int, center: Tuple[float, float] = (0, 0), scale: float = 1.0):
    return (np.random.rand(num_v, 2) * 2 - 1) * scale + center


def draw_line_edge(
    ax: Axes,
    v_coor: np.array,
    v_size: list,
    e_list: List[Tuple[int, int]],
    show_arrow: bool,
    e_color: list,
    e_line_width: list,
):
    arrow_head_width = (
        [0.015 * w for w in e_line_width] if show_arrow else [0] * len(e_list)
    )

    for eidx, e in enumerate(e_list):
        start_pos = v_coor[e[0]]
        end_pos = v_coor[e[1]]

        dir = end_pos - start_pos
        dir = dir / np.linalg.norm(dir)

        start_pos = start_pos + dir * v_size[e[0]]
        end_pos = end_pos - dir * v_size[e[1]]

        x, y = start_pos[0], start_pos[1]
        dx, dy = end_pos[0] - x, end_pos[1] - y

        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=arrow_head_width[eidx],
            color=e_color[eidx],
            linewidth=e_line_width[eidx],
            length_includes_head=True,
        )


def draw_circle_edge(
    ax: Axes,
    v_coor: List[Tuple[float, float]],
    v_size: list,
    e_list: List[Tuple[int, int]],
    e_color: list,
    e_fill_color: list,
    e_line_width: list,
):
    n_v = len(v_coor)
    line_paths, arc_paths, vertices = hull_layout(n_v, e_list, v_coor, v_size)
    for eidx, lines in enumerate(line_paths):
        pathdata = []
        for line in lines:
            if len(line) == 0:
                continue
            start_pos, end_pos = line
            pathdata.append((Path.MOVETO, start_pos.tolist()))
            pathdata.append((Path.LINETO, end_pos.tolist()))

        if len(list(zip(*pathdata))) == 0:
            continue
        codes, verts = zip(*pathdata)
        path = Path(verts, codes)

        ax.add_patch(
            PathPatch(
                path,
                linewidth=e_line_width[eidx],
                facecolor=e_fill_color[eidx],
                edgecolor=e_color[eidx],
            )
        )

    for eidx, arcs in enumerate(arc_paths):
        for arc in arcs:
            center, theta1, theta2, radius = arc
            x, y = center[0], center[1]

            patcjes_arc = matplotlib.patches.Arc(
                (x, y),
                2 * radius,
                2 * radius,
                theta1=theta1,
                theta2=theta2,
                color=e_color[eidx],
                linewidth=e_line_width[eidx],
                # edgecolor=e_color[eidx],
                edgecolor=e_color[eidx],
                facecolor=e_fill_color[eidx],
            )

            ax.add_patch(
                matplotlib.patches.Arc(
                    (x, y),
                    2 * radius,
                    2 * radius,
                    theta1=theta1,
                    theta2=theta2,
                    color=e_color[eidx],
                    linewidth=e_line_width[eidx],
                    # edgecolor=e_color[eidx],
                    edgecolor=e_color[eidx],
                    facecolor=e_fill_color[eidx],
                )
            )


def edge_list_to_incidence_matrix(num_v: int, e_list: List[tuple]) -> np.ndarray:
    v_idx = list(chain(*e_list))
    e_idx = [[idx] * len(e) for idx, e in enumerate(e_list)]
    e_idx = list(chain(*e_idx))
    H = np.zeros((num_v, len(e_list)))
    H[v_idx, e_idx] = 1
    return H


def draw_vertex(
    ax: Axes,
    v_coor: List[Tuple[float, float]],
    v_label: Optional[List[str]],
    font_size: int,
    font_family: str,
    v_size: list,
    v_color: list,
    edgecolors,
    v_line_width: list,
):
    patches = []
    n = v_coor.shape[0]
    if v_label is None:
        v_label = [""] * n
    for coor, label, size, width in zip(v_coor.tolist(), v_label, v_size, v_line_width):
        circle = Circle(coor, size)
        circle.lineWidth = width
        # circle.label = label
        if label != "":
            x, y = coor[0], coor[1]
            offset = 0, -1.3 * size
            x += offset[0]
            y += offset[1]
            ax.text(
                x,
                y,
                label,
                fontsize=font_size,
                fontfamily=font_family,
                ha="center",
                va="top",
            )
        patches.append(circle)
    edgecolors = "black" if edgecolors == None else edgecolors
    p = PatchCollection(patches, facecolors=v_color, edgecolors=edgecolors)
    ax.add_collection(p)


def hull_layout(n_v, e_list, pos, v_size, radius_increment=0.3):
    line_paths = [None] * len(e_list)
    arc_paths = [None] * len(e_list)

    polygons_vertices_index = []
    vertices_radius = np.array(v_size)
    vertices_increased_radius = vertices_radius * radius_increment
    vertices_radius += vertices_increased_radius

    e_degree = [len(e) for e in e_list]
    e_idxs = np.argsort(np.array(e_degree))

    # for edge in e_list:
    for e_idx in e_idxs:
        edge = list(e_list[e_idx])

        line_path_for_e = []
        arc_path_for_e = []

        if len(edge) == 1:
            arc_path_for_e.append([pos[edge[0]], 0, 360, vertices_radius[edge[0]]])

            vertices_radius[edge] += vertices_increased_radius[edge]

            line_paths[e_idx] = line_path_for_e
            arc_paths[e_idx] = arc_path_for_e
            continue

        pos_in_edge = pos[edge]
        if len(edge) == 2:
            vertices_index = np.array((0, 1), dtype=np.int64)
        else:
            hull = ConvexHull(pos_in_edge)
            vertices_index = hull.vertices

        n_vertices = vertices_index.shape[0]

        vertices_index = np.append(vertices_index, vertices_index[0])  # close the loop

        thetas = []

        for i in range(n_vertices):
            # line
            i1 = edge[vertices_index[i]]
            i2 = edge[vertices_index[i + 1]]

            r1 = vertices_radius[i1]
            r2 = vertices_radius[i2]

            p1 = pos[i1]
            p2 = pos[i2]

            dp = p2 - p1
            dp_len = vlen(dp)

            beta = radian_from_atan(dp[0], dp[1])
            alpha = common_tangent_radian(r1, r2, dp_len)

            theta = beta - alpha
            start_point = polar_position(r1, theta, p1)
            end_point = polar_position(r2, theta, p2)

            line_path_for_e.append((start_point, end_point))
            thetas.append(theta)

        for i in range(n_vertices):
            # arcs
            theta_1 = thetas[i - 1]
            theta_2 = thetas[i]

            arc_center = pos[edge[vertices_index[i]]]
            radius = vertices_radius[edge[vertices_index[i]]]

            theta_1, theta_2 = rad_2_deg(theta_1), rad_2_deg(theta_2)
            arc_path_for_e.append((arc_center, theta_1, theta_2, radius))

        vertices_radius[edge] += vertices_increased_radius[edge]

        polygons_vertices_index.append(vertices_index.copy())

        # line_paths.append(line_path_for_e)
        # arc_paths.append(arc_path_for_e)
        line_paths[e_idx] = line_path_for_e
        arc_paths[e_idx] = arc_path_for_e

    return line_paths, arc_paths, polygons_vertices_index


def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    """Apply an alpha (or list of alphas) to the colors provided.

    Parameters
    ----------

    colors : color string or array of floats (default='r')
        Color of element. Can be a single color format string,
        or a sequence of colors with the same length as nodelist.
        If numeric values are specified they will be mapped to
        colors using the cmap and vmin,vmax parameters.  See
        matplotlib.scatter for more details.

    alpha : float or array of floats
        Alpha values for elements. This can be a single alpha value, in
        which case it will be applied to all the elements of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    elem_list : array of networkx objects
        The list of elements which are being colored. These could be nodes,
        edges or labels.

    cmap : matplotlib colormap
        Color map for use if colors is a list of floats corresponding to points
        on a color mapping.

    vmin, vmax : float
        Minimum and maximum values for normalizing colors if a colormap is used

    Returns
    -------

    rgba_colors : numpy ndarray
        Array containing RGBA format values for each of the node colours.

    """
    from itertools import cycle
    from itertools import islice
    from numbers import Number

    import matplotlib as mpl
    import matplotlib.cm  # call as mpl.cm
    import matplotlib.colors  # call as mpl.colors
    import numpy as np

    # If we have been provided with a list of numbers as long as elem_list,
    # apply the color mapping.
    if len(colors) == len(elem_list) and isinstance(colors[0], Number):
        mapper = mpl.cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_colors = mapper.to_rgba(colors)
    # Otherwise, convert colors to matplotlib's RGB using the colorConverter
    # object.  These are converted to numpy ndarrays to be consistent with the
    # to_rgba method of ScalarMappable.
    else:
        try:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(colors)])
        except ValueError:
            rgba_colors = np.array(
                [mpl.colors.colorConverter.to_rgba(color) for color in colors]
            )
    # Set the final column of the rgba_colors to have the relevant alpha values
    try:
        # If alpha is longer than the number of colors, resize to the number of
        # elements.  Also, if rgba_colors.size (the number of elements of
        # rgba_colors) is the same as the number of elements, resize the array,
        # to avoid it being interpreted as a colormap by scatter()
        if len(alpha) > len(rgba_colors) or rgba_colors.size == len(elem_list):
            rgba_colors = np.resize(rgba_colors, (len(elem_list), 4))
            rgba_colors[1:, 0] = rgba_colors[0, 0]
            rgba_colors[1:, 1] = rgba_colors[0, 1]
            rgba_colors[1:, 2] = rgba_colors[0, 2]
        rgba_colors[:, 3] = list(islice(cycle(alpha), len(rgba_colors)))
    except TypeError:
        rgba_colors[:, -1] = alpha
    return rgba_colors


# def draw_easygraph_nodes(
#     G,
#     pos,
#     nodelist=None,
#     node_size=300,
#     node_color="#1f78b4",
#     node_shape="o",
#     alpha=None,
#     cmap=None,
#     vmin=None,
#     vmax=None,
#     ax=None,
#     linewidths=None,
#     edgecolors=None,
#     label=None,
#     margins=None,
# ):
#     """Draw the nodes of the graph G.

#     This draws only the nodes of the graph G.

#     Parameters
#     ----------
#     G : graph
#         A easygraph graph

#     pos : dictionary
#         A dictionary with nodes as keys and positions as values.
#         Positions should be sequences of length 2.

#     ax : Matplotlib Axes object, optional
#         Draw the graph in the specified Matplotlib axes.

#     nodelist : list (default list(G))
#         Draw only specified nodes

#     node_size : scalar or array (default=300)
#         Size of nodes.  If an array it must be the same length as nodelist.

#     node_color : color or array of colors (default='#1f78b4')
#         Node color. Can be a single color or a sequence of colors with the same
#         length as nodelist. Color can be string or rgb (or rgba) tuple of
#         floats from 0-1. If numeric values are specified they will be
#         mapped to colors using the cmap and vmin,vmax parameters. See
#         matplotlib.scatter for more details.

#     node_shape :  string (default='o')
#         The shape of the node.  Specification is as matplotlib.scatter
#         marker, one of 'so^>v<dph8'.

#     alpha : float or array of floats (default=None)
#         The node transparency.  This can be a single alpha value,
#         in which case it will be applied to all the nodes of color. Otherwise,
#         if it is an array, the elements of alpha will be applied to the colors
#         in order (cycling through alpha multiple times if necessary).

#     cmap : Matplotlib colormap (default=None)
#         Colormap for mapping intensities of nodes

#     vmin,vmax : floats or None (default=None)
#         Minimum and maximum for node colormap scaling

#     linewidths : [None | scalar | sequence] (default=1.0)
#         Line width of symbol border

#     edgecolors : [None | scalar | sequence] (default = node_color)
#         Colors of node borders

#     label : [None | string]
#         Label for legend

#     margins : float or 2-tuple, optional
#         Sets the padding for axis autoscaling. Increase margin to prevent
#         clipping for nodes that are near the edges of an image. Values should
#         be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`
#         for details. The default is `None`, which uses the Matplotlib default.

#     Returns
#     -------
#     matplotlib.collections.PathCollection
#         `PathCollection` of the nodes.

#     Examples
#     --------
#     >>> from easygraph.datasets import get_graph_karateclub
#     >>> import easygraph as eg
#     >>> G = get_graph_karateclub()
#     >>> nodes = eg.draw_easygraph_nodes(G, pos=eg.circular_position(G))


#     """
#     from collections.abc import Iterable

#     import matplotlib as mpl
#     import matplotlib.collections  # call as mpl.collections
#     import matplotlib.pyplot as plt
#     import numpy as np

#     if ax is None:
#         ax = plt.gca()

#     if nodelist is None:
#         nodelist = list(G)

#     if len(nodelist) == 0:  # empty nodelist, no drawing
#         return mpl.collections.PathCollection(None)

#     try:
#         xy = np.asarray([pos[v] for v in nodelist])
#     except KeyError as err:
#         raise EasyGraphError(f"Node {err} has no position.") from err

#     if isinstance(alpha, Iterable):
#         node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
#         alpha = None

#     node_collection = ax.scatter(
#         xy[:, 0],
#         xy[:, 1],
#         s=node_size,
#         c=node_color,
#         marker=node_shape,
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#         alpha=alpha,
#         linewidths=linewidths,
#         edgecolors=edgecolors,
#         label=label,
#     )
#     ax.tick_params(
#         axis="both",
#         which="both",
#         bottom=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False,
#     )

#     if margins is not None:
#         if isinstance(margins, Iterable):
#             ax.margins(*margins)
#         else:
#             ax.margins(margins)

#     node_collection.set_zorder(2)
#     return node_collection


# def draw_curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity='random'):
#     # Get nodes into np array
#     edges = np.array(G.edges())
#     l = edges.shape[0]

#     if polarity == 'random':
#         # Random polarity of curve
#         rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
#     else:
#         # Create a fixed (hashed) polarity column in the case we use fixed polarity
#         # This is useful, e.g., for animations
#         rnd = np.where(np.mod(np.vectorize(hash)(edges[:,0])+np.vectorize(hash)(edges[:,1]),2)==0,-1,1)

#     # Coordinates (x,y) of both nodes for each edge
#     # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
#     # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
#     u, inv = np.unique(edges, return_inverse = True)
#     coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
#     coords_node1 = coords[:,0,:]
#     coords_node2 = coords[:,1,:]

#     # Swap node1/node2 allocations to make sure the directionality works correctly
#     should_swap = coords_node1[:,0] > coords_node2[:,0]
#     coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]

#     # Distance for control points
#     dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

#     # Gradients of line connecting node & perpendicular
#     m1 = (coords_node2[:,1]-coords_node1[:,1])/(coords_node2[:,0]-coords_node1[:,0])
#     m2 = -1/m1

#     # Temporary points along the line which connects two nodes
#     # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
#     t1 = dist/np.sqrt(1+m1**2)
#     v1 = np.array([np.ones(l),m1])
#     coords_node1_displace = coords_node1 + (v1*t1).T
#     coords_node2_displace = coords_node2 - (v1*t1).T

#     # Control points, same distance but along perpendicular line
#     # rnd gives the 'polarity' to determine which side of the line the curve should arc
#     t2 = dist/np.sqrt(1+m2**2)
#     v2 = np.array([np.ones(len(edges)),m2])
#     coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
#     coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

#     # Combine all these four (x,y) columns into a 'node matrix'
#     node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

#     # Create the Bezier curves and store them in a list
#     curveplots = []
#     for i in range(l):
#         nodes = node_matrix[:,i,:].T
#         curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0,1,bezier_precision)).T)
#     # Return an array of these curves
#     curves = np.array(curveplots)
#     return curves

# def draw_curved_graph(G, colors, ax):
#     #G = to_networkx(G)
#     # layout
#     pos = eg.spring_layout(G, iterations=50)
#     eg.draw_networkx_nodes(G, pos, ax=ax, node_size=200, node_color=colors[0], alpha=0.5)

#     # 绘制标签
#     eg.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_family='Arial', font_color='black')

#     # Produce the curves
#     curves = draw_curved_edges(G, pos)
#     lc = LineCollection(curves, color=colors[1], alpha=0.4)

#     # 添加连线
#     ax.add_collection(lc)

#     # 设置坐标轴参数
#     ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

#     plt.savefig('Figure.pdf')
#     plt.show()
