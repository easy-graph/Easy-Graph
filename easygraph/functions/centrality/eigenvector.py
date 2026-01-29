import math
import easygraph as eg
from easygraph.utils import *
from easygraph.utils.decorators import *
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
from collections import defaultdict

__all__ = ["eigenvector_centrality"]

@not_implemented_for("multigraph")
@hybrid("cpp_eigenvector_centrality")
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight=None):
    """Calculate eigenvector centrality for nodes in the graph
    
    Eigenvector centrality is based on the idea that a node's importance
    depends on the importance of its neighboring nodes.
    Specifically, a node's centrality is proportional to the sum of 
    centrality values of its neighbors.
    
    Parameters
    ----------
    G : graph object
        An undirected or directed graph
    
    max_iter : int, optional (default=100)
        Maximum number of iterations for the power method
    
    tol : float, optional (default=1.0e-6)
        Convergence threshold; algorithm terminates when the difference
        between centrality values in consecutive iterations is less than this value
    
    nstart : dictionary, optional (default=None)
        Dictionary mapping nodes to initial centrality values
        If None, the ARPACK solver is used to directly compute the eigenvector
    
    weight : string or None, optional (default=None)
        Name of the edge attribute to be used as edge weight
        If None, all edges are considered to have weight 1
    
    Returns
    -------
    centrality : dictionary
        Dictionary mapping nodes to their eigenvector centrality values
    
    Raises
    ------
    EasyGraphPointlessConcept
        When input is an empty graph
    
    EasyGraphError
        When the algorithm fails to converge within the specified maximum iterations
    
    Notes
    -----
    This algorithm uses the power iteration method to find the principal eigenvector.
    When nstart is not provided, the ARPACK solver is used for efficiency.
    The returned centrality values are normalized.
    """
    
    if len(G) == 0:
        raise eg.EasyGraphPointlessConcept(
            "cannot compute centrality for the null graph"
        )

    if len(G) == 1:
        raise eg.EasyGraphPointlessConcept(
            "cannot compute eigenvector centrality for a single node graph"
        )
    
    
    # Build node list and mapping
    nodelist = list(G.nodes)
    n = len(nodelist)
    node_map = {node: i for i, node in enumerate(nodelist)}
    
    # Build weighted adjacency matrix
    row, col, data = [], [], []
    for u in nodelist:
        u_idx = node_map[u]
        for v, attrs in G[u].items():
            if v in node_map:
                v_idx = node_map[v]
                w = attrs.get(weight, 1.0) if weight else 1.0
                # Build transpose matrix for centrality calculation
                row.append(v_idx)
                col.append(u_idx)
                data.append(float(w))
    
    # Create CSR format sparse matrix
    A = sparse.csr_matrix((data, (row, col)), shape=(n, n))

    # Detect and handle isolated nodes
    row_sums = np.array(A.sum(axis=1)).flatten()
    col_sums = np.array(A.sum(axis=0)).flatten()
    isolated_nodes = np.where((row_sums == 0) & (col_sums == 0))[0]
    
    has_isolated = len(isolated_nodes) > 0
    isolated_indices = []
    
    # Add small self-loops to isolated nodes for stability
    if has_isolated:
        # Store isolated node indices
        isolated_indices = isolated_nodes.tolist()
        
        # Add small self-loop weights to isolated nodes
        for idx in isolated_indices:
            A[idx, idx] = 1.0e-4  # Small enough to not affect results, but maintains numerical stability
    if nstart is not None:
        # Use custom initial vector for power iteration
        v = np.array([nstart.get(n, 1.0) for n in nodelist], dtype=float)
        v = v / np.sum(np.abs(v))
        
        # Power iteration method to compute principal eigenvector
        v_last = np.zeros_like(v)
        for _ in range(max_iter):
            np.copyto(v_last, v)
            v = A @ v_last  # Sparse matrix multiplication
            
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                v = v_last.copy()
                break
            v = v / norm  # Normalization
            
            # Check convergence
            if np.linalg.norm(v - v_last) < tol:
                break
        else:
            raise eg.EasyGraphError(f"Eigenvector calculation did not converge in {max_iter} iterations")
        
        centrality = v
    else:
        # Use ARPACK solver to directly compute the principal eigenvector
        eigenvalues, eigenvectors = linalg.eigs(A, k=1, which='LR', 
                                              maxiter=max_iter, tol=tol)
        centrality = np.real(eigenvectors[:,0])
    
    # Ensure positive results and normalize
    if centrality.sum() < 0:
        centrality = -centrality
    
    centrality = centrality / np.linalg.norm(centrality)
     # Set centrality of isolated nodes to zero
    if has_isolated:
        for idx in isolated_indices:
            centrality[idx] = 0.0
        # Renormalize if needed
        if np.sum(centrality) > 0:
            centrality = centrality / np.linalg.norm(centrality)
    
    # Return dictionary of node centrality values
    return {nodelist[i]: float(centrality[i]) for i in range(n)}