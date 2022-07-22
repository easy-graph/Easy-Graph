"""
**************
UCINET DL
**************
Read and write graphs in UCINET DL format.
This implementation currently supports only the 'fullmatrix' data format.
Format
------
The UCINET DL format is the most common file format used by UCINET package.
Basic example:
DL N = 5
Data:
0 1 1 1 1
1 0 1 0 0
1 1 0 0 1
1 0 0 0 0
1 0 1 0 0
References
----------
    See UCINET User Guide or http://www.analytictech.com/ucinet/help/hs5000.htm
    for full format information. Short version on http://www.analytictech.com/networks/dataentry.htm
"""


import re
import shlex

import easygraph as eg
import numpy as np

from easygraph.utils import open_file


__all__ = ["generate_ucinet", "read_ucinet", "parse_ucinet", "write_ucinet"]


def generate_ucinet(G):
    """Generate lines in UCINET graph format.
    Parameters
    ----------
    G : graph
       A EasyGraph graph
    Examples
    --------
    Notes
    -----
    The default format 'fullmatrix' is used (for UCINET DL format).

    References
    ----------
    See UCINET User Guide or http://www.analytictech.com/ucinet/help/hs5000.htm
    for full format information. Short version on http://www.analytictech.com/networks/dataentry.htm
    """

    n = G.number_of_nodes()
    nodes = sorted(list(G.nodes))
    yield "dl n=%i format=fullmatrix" % n

    # Labels
    try:
        int(nodes[0])
    except ValueError:
        s = "labels:\n"
        for label in nodes:
            s += label + " "
        yield s

    yield "data:"

    yield str(np.asmatrix(eg.to_numpy_array(G, nodelist=nodes, dtype=int))).replace(
        "[", " "
    ).replace("]", " ").lstrip().rstrip()


@open_file(0, mode="rb")
def read_ucinet(path, encoding="UTF-8"):
    """Read graph in UCINET format from path.
    Parameters
    ----------
    path : file or string
       File or filename to read.
       Filenames ending in .gz or .bz2 will be uncompressed.
    Returns
    -------
    G : EasyGraph MultiGraph or MultiDiGraph.
    Examples
    --------
    >>> G=eg.path_graph(4)
    >>> eg.write_ucinet(G, "test.dl")
    >>> G=eg.read_ucinet("test.dl")
    To create a Graph instead of a MultiGraph use
    >>> G1=eg.Graph(G)
    See Also
    --------
    parse_ucinet()
    References
    ----------
    See UCINET User Guide or http://www.analytictech.com/ucinet/help/hs5000.htm
    for full format information. Short version on http://www.analytictech.com/networks/dataentry.htm
    """
    lines = (line.decode(encoding) for line in path)
    return parse_ucinet(lines)


@open_file(1, mode="wb")
def write_ucinet(G, path, encoding="UTF-8"):
    """Write graph in UCINET format to path.
    Parameters
    ----------
    G : graph
       A EasyGraph graph
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.
    Examples
    --------
    >>> G=eg.path_graph(4)
    >>> eg.write_ucinet(G, "test.net")
    References
    ----------
    See UCINET User Guide or http://www.analytictech.com/ucinet/help/hs5000.htm
    for full format information. Short version on http://www.analytictech.com/networks/dataentry.htm
    """
    for line in generate_ucinet(G):
        line += "\n"
        path.write(line.encode(encoding))


def parse_ucinet(lines):
    """Parse UCINET format graph from string or iterable.
    Currently only the 'fullmatrix', 'nodelist1' and 'nodelist1b' formats are supported.
    Parameters
    ----------
    lines : string or iterable
       Data in UCINET format.
    Returns
    -------
    G : EasyGraph graph
    See Also
    --------
    read_ucinet()
    References
    ----------
    See UCINET User Guide or http://www.analytictech.com/ucinet/help/hs5000.htm
    for full format information. Short version on http://www.analytictech.com/networks/dataentry.htm
    """
    from numpy import genfromtxt
    from numpy import isnan
    from numpy import reshape

    G = eg.MultiDiGraph()

    if not isinstance(lines, str):
        s = ""
        for line in lines:
            if type(line) == bytes:
                s += line.decode("utf-8")
            else:
                s += line
        lines = s
    lexer = shlex.shlex(lines.lower())
    lexer.whitespace += ",="
    lexer.whitespace_split = True

    number_of_nodes = 0
    number_of_matrices = 0
    nr = 0  # number of rows (rectangular matrix)
    nc = 0  # number of columns (rectangular matrix)
    ucinet_format = "fullmatrix"  # Format by default
    labels = {}  # Contains labels of nodes
    row_labels_embedded = False  # Whether labels are embedded in data or not
    cols_labels_embedded = False
    diagonal = True  # whether the main diagonal is present or absent

    KEYWORDS = ("format", "data:", "labels:")  # TODO remove ':' in keywords

    while lexer:
        try:
            token = next(lexer)
        except StopIteration:
            break
        # print "Token : %s" % token
        if token.startswith("n"):
            if token.startswith("nr"):
                nr = int(get_param(r"\d+", token, lexer))
                number_of_nodes = max(nr, nc)
            elif token.startswith("nc"):
                nc = int(get_param(r"\d+", token, lexer))
                number_of_nodes = max(nr, nc)
            elif token.startswith("nm"):
                number_of_matrices = int(get_param(r"\d+", token, lexer))
            else:
                number_of_nodes = int(get_param(r"\d+", token, lexer))
                nr = number_of_nodes
                nc = number_of_nodes

        elif token.startswith("diagonal"):
            diagonal = get_param("present|absent", token, lexer)

        elif token.startswith("format"):
            ucinet_format = get_param(
                """^(fullmatrix|upperhalf|lowerhalf|nodelist1|nodelist2|nodelist1b|\
edgelist1|edgelist2|blockmatrix|partition)$""",
                token,
                lexer,
            )

        # TODO : row and columns labels
        elif token.startswith("row"):  # Row labels
            pass
        elif token.startswith("column"):  # Columns labels
            pass

        elif token.startswith("labels"):
            token = next(lexer)
            i = 0
            while token not in KEYWORDS:
                if token.startswith("embedded"):
                    row_labels_embedded = True
                    cols_labels_embedded = True
                    break
                else:
                    labels[i] = token.replace(
                        '"', ""
                    )  # for labels with embedded spaces
                    i += 1
                    try:
                        token = next(lexer)
                    except StopIteration:
                        break
        elif token.startswith("data"):
            break

    data_lines = lines.lower().split("data:", 1)[1]
    # Generate edges
    params = {}
    if cols_labels_embedded:
        # params['names'] = True
        labels = dict(zip(range(0, nc), data_lines.splitlines()[1].split()))
        # params['skip_header'] = 2  # First character is \n
    if row_labels_embedded:  # Skip first column
        # TODO rectangular case : labels can differ from rows to columns
        # params['usecols'] = range(1, nc + 1)
        pass

    if ucinet_format == "fullmatrix":
        # In Python3 genfromtxt requires bytes string
        try:
            data_lines = bytes(data_lines, "utf-8")
        except TypeError:
            pass
        # Do not use splitlines() because it is not necessarily written as a square matrix
        data = genfromtxt([data_lines], case_sensitive=False, **params)
        if cols_labels_embedded or row_labels_embedded:
            # data = insert(data, 0, float('nan'))
            data = data[~isnan(data)]
        mat = reshape(data, (max(number_of_nodes, nr), -1))
        G = eg.from_numpy_array(mat, create_using=eg.MultiDiGraph())

    elif ucinet_format in (
        "nodelist1",
        "nodelist1b",
    ):  # Since genfromtxt only accepts square matrix...
        s = ""
        for i, line in enumerate(data_lines.splitlines()):
            row = line.split()
            if row:
                if ucinet_format == "nodelist1b" and row[0] == "0":
                    pass
                else:
                    for neighbor in row[1:]:
                        if ucinet_format == "nodelist1":
                            source = row[0]
                        else:
                            source = str(i)
                        s += source + " " + neighbor + "\n"

        G = eg.parse_edgelist(
            s.splitlines(),
            nodetype=str if row_labels_embedded and cols_labels_embedded else int,
            create_using=eg.MultiDiGraph(),
        )

        if not row_labels_embedded or not cols_labels_embedded:
            G = eg.relabel_nodes(G, dict(zip(list(G.nodes), [i - 1 for i in G.nodes])))

    elif ucinet_format == "edgelist1":
        G = eg.parse_edgelist(
            data_lines.splitlines(),
            nodetype=str if row_labels_embedded and cols_labels_embedded else int,
            create_using=eg.MultiDiGraph(),
        )

        if not row_labels_embedded or not cols_labels_embedded:
            G = eg.relabel_nodes(G, dict(zip(list(G.nodes), [i - 1 for i in G.nodes])))

    # Relabel nodes
    if labels:
        try:
            if len(list(G.nodes)) < number_of_nodes:
                G.add_nodes_from(
                    labels.values() if labels else range(0, number_of_nodes)
                )
            G = eg.relabel_nodes(G, labels)
        except KeyError:
            pass  # Nodes already labelled

    return G


def get_param(regex, token, lines):
    """
    Get a parameter value in UCINET DL file
    :param regex: string with the regex matching the parameter value
    :param token: token (string) in which we search for the parameter
    :param lines: to iterate through the next tokens
    :return:
    """
    n = token
    query = re.search(regex, n)
    while query is None:
        try:
            n = next(lines)
        except StopIteration:
            raise Exception("Parameter %s value not recognized" % token)
        query = re.search(regex, n)
    return query.group()
