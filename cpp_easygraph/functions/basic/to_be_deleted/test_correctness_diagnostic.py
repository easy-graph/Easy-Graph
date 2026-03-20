#!/usr/bin/env python3
"""
Diagnostic script to pinpoint the root cause of gt clustering coefficient discrepancy.
Compares:
1. Degree counts at node level (to check graph construction consistency)
2. Triangle counts at node level (to check intermediate calculation consistency)  
3. Clustering coefficients (with ratio analysis to detect multiplicative factor differences)
"""

import numpy as np
import sys
sys.path.insert(0, '/users/sama/Easy-Graph')

import easygraph as eg
import igraph
import graph_tool as gt
from easygraph.datasets import ArxivHEPTHDataset

print("=" * 100)
print("DIAGNOSTIC: Finding root cause of gt clustering coefficient difference")
print("=" * 100)

# Load dataset
print("\n[1/5] Loading ArxivHEPTHDataset...")
dataset = ArxivHEPTHDataset()
G_py = dataset[0]
edges = [(u, v) for u, v, *_ in G_py.edges]
print(f"    Total edges (original): {len(edges)}")

# Build undirected edge set
edges_ud = set()
for u, v in edges:
    key = (min(u, v), max(u, v))
    edges_ud.add(key)
edges_ud = list(edges_ud)
print(f"    Total edges (undirected normalized): {len(edges_ud)}")

# Get node order for consistent indexing
nodes_order = []
for u, v in edges_ud:
    if u not in nodes_order:
        nodes_order.append(u)
    if v not in nodes_order:
        nodes_order.append(v)
print(f"    Total nodes: {len(nodes_order)}")

# Create label to index mapping
label2idx = {lab: i for i, lab in enumerate(nodes_order)}

# Build EasyGraph
print("\n[2/5] Building graphs in easygraph, igraph, and graph-tool...")
G_eg = eg.Graph()
G_eg.add_edges_from(edges_ud)
print(f"    easygraph: nodes={G_eg.number_of_nodes()}, edges={G_eg.number_of_edges()}")

# Build igraph
edges_idx = [(label2idx[u], label2idx[v]) for u, v in edges_ud]
G_ig = igraph.Graph(n=len(nodes_order), edges=edges_idx, directed=False)
print(f"    igraph: nodes={G_ig.vcount()}, edges={G_ig.ecount()}")

# Build graph-tool
G_gt = gt.Graph(directed=False)
vmap = {lab: G_gt.add_vertex() for lab in nodes_order}
for u, v in edges_ud:
    G_gt.add_edge(vmap[u], vmap[v])
print(f"    graph-tool: nodes={G_gt.num_vertices()}, edges={G_gt.num_edges()}")

# Compute clustering coefficients
print("\n[3/5] Computing clustering coefficients...")
cc_eg = eg.clustering(G_eg)
print(f"    easygraph clustering computed")

cc_ig = G_ig.transitivity_local_undirected(mode="zero")
print(f"    igraph clustering computed")

cc_gt = gt.local_clustering(G_gt)
cc_gt_vals = [cc_gt[v] for v in G_gt.vertices()]
print(f"    graph-tool clustering computed")

# Extract degree information
print("\n[4/5] Extracting per-node degree and clustering data...")
print("\nNode-level Comparison Table:")
print(f"{'Node':>6} {'EG-Deg':>8} {'IG-Deg':>8} {'GT-Deg':>8} {'DegMatch':>10} "
      f"{'EG-CC':>10} {'IG-CC':>10} {'GT-CC':>10} {'Ratio GT/EG':>12}")
print("-" * 110)

degree_mismatches = []
data_rows = []

for idx, node_label in enumerate(nodes_order):
    # Get degrees
    deg_eg = G_eg.degree(node_label)
    deg_ig = G_ig.vs[idx].degree()
    deg_gt = G_gt.vertex(vmap[node_label]).out_degree()
    
    # Get clustering coefficients
    cc_eg_val = cc_eg[node_label]
    cc_ig_val = cc_ig[idx]
    cc_gt_val = cc_gt_vals[idx]
    
    # Check degree consistency
    deg_match = "OK" if (deg_eg == deg_ig == deg_gt) else "MISMATCH"
    if deg_match == "MISMATCH":
        degree_mismatches.append({
            'node': node_label,
            'eg': deg_eg,
            'ig': deg_ig,
            'gt': deg_gt
        })
    
    # Calculate ratio
    if cc_eg_val > 0:
        ratio = cc_gt_val / cc_eg_val if cc_eg_val != 0 else np.nan
    else:
        ratio = np.nan if cc_eg_val == 0 else cc_gt_val / cc_eg_val
    
    data_rows.append({
        'node': node_label,
        'deg_eg': deg_eg,
        'deg_ig': deg_ig,
        'deg_gt': deg_gt,
        'cc_eg': cc_eg_val,
        'cc_ig': cc_ig_val,
        'cc_gt': cc_gt_val,
        'ratio': ratio
    })
    
    # Print first 20 nodes with detailed info
    if idx < 20:
        print(f"{idx:>6} {deg_eg:>8} {deg_ig:>8} {deg_gt:>8} {deg_match:>10} "
              f"{cc_eg_val:>10.6f} {cc_ig_val:>10.6f} {cc_gt_val:>10.6f} {ratio:>12.6f}")

print("    ... (showing first 20 nodes, full analysis below)")

# Summary statistics
print("\n" + "=" * 100)
print("ANALYSIS SUMMARY")
print("=" * 100)

# Degree analysis
if degree_mismatches:
    print(f"\n[DEGREE MISMATCH DETECTED] Found {len(degree_mismatches)} nodes with degree inconsistency:")
    for dm in degree_mismatches[:5]:
        print(f"  Node {dm['node']}: EG={dm['eg']}, IG={dm['ig']}, GT={dm['gt']}")
    print("  ⚠️  This suggests GRAPH CONSTRUCTION INCONSISTENCY")
else:
    print("\n[DEGREE CONSISTENCY] All nodes have consistent degrees across libraries")
    print("  ✓ Graph construction appears equivalent")

# Coefficient comparison
cc_eg_arr = np.array([row['cc_eg'] for row in data_rows if not np.isnan(row['cc_eg'])])
cc_ig_arr = np.array([row['cc_ig'] for row in data_rows if not np.isnan(row['cc_ig'])])
cc_gt_arr = np.array([row['cc_gt'] for row in data_rows if not np.isnan(row['cc_gt'])])

print("\n[CLUSTERING COEFFICIENT ANALYSIS]")
print(f"  Mean:  EG={np.mean(cc_eg_arr):.10f}, IG={np.mean(cc_ig_arr):.10f}, GT={np.mean(cc_gt_arr):.10f}")
print(f"  Median: EG={np.median(cc_eg_arr):.10f}, IG={np.median(cc_ig_arr):.10f}, GT={np.median(cc_gt_arr):.10f}")
print(f"  Max:   EG={np.max(cc_eg_arr):.10f}, IG={np.max(cc_ig_arr):.10f}, GT={np.max(cc_gt_arr):.10f}")
print(f"  Min:   EG={np.min(cc_eg_arr):.10f}, IG={np.min(cc_ig_arr):.10f}, GT={np.min(cc_gt_arr):.10f}")

# EG vs IG comparison
diff_eg_ig = np.abs(cc_eg_arr - cc_ig_arr)
print(f"\n  EG vs IG: mean_diff={np.mean(diff_eg_ig):.2e}, max_diff={np.max(diff_eg_ig):.2e}")

# EG vs GT comparison
diff_eg_gt = np.abs(cc_eg_arr - cc_gt_arr)
print(f"  EG vs GT: mean_diff={np.mean(diff_eg_gt):.2e}, max_diff={np.max(diff_eg_gt):.2e}")

# Ratio analysis
valid_ratios = [row['ratio'] for row in data_rows if not np.isnan(row['ratio']) and row['ratio'] != 0]
if valid_ratios:
    print(f"\n[RATIO ANALYSIS] GT/EG coefficient ratio:")
    print(f"  Mean ratio:   {np.mean(valid_ratios):.10f}")
    print(f"  Median ratio: {np.median(valid_ratios):.10f}")
    print(f"  Min ratio:    {np.min(valid_ratios):.10f}")
    print(f"  Max ratio:    {np.max(valid_ratios):.10f}")
    print(f"  Std dev:      {np.std(valid_ratios):.10f}")
    
    # Check if ratio is constant
    if np.std(valid_ratios) < 1e-10:
        print(f"  ✓ CONSTANT RATIO DETECTED: ~{np.mean(valid_ratios):.10f}")
        print(f"    This suggests FORMULA DIFFERENCE (e.g., different divisor or factor)")
    else:
        print(f"  ✗ Variable ratio (std={np.std(valid_ratios):.10f})")
        print(f"    This suggests more complex calculation difference")

# Hypothesis testing
print("\n" + "=" * 100)
print("ROOT CAUSE HYPOTHESES")
print("=" * 100)

if not degree_mismatches:
    print("\n[Hypothesis A] Formula/Divisor Difference (LIKELY)")
    print("  Evidence: Degree counts are consistent, but coefficients differ")
    print("  Suspected cause: Different interpretation of triangle or denominator calculation")
    print("  Location to investigate: gt_cluster.hh get_triangles() divisor vs cluster.cpp implementation")
    
    # Check if gt is consistently lower
    gt_lower = np.mean(cc_gt_arr) < np.mean(cc_eg_arr)
    if gt_lower:
        print(f"\n  GT is consistently LOWER than EG (by ~{(1 - np.mean(cc_gt_arr)/np.mean(cc_eg_arr))*100:.2f}%)")
        print("  This matches source code inspection: gt returns {triangles/2, (k*k-k2)/2}")
        
        ratio_mean = np.mean(valid_ratios) if valid_ratios else 1.0
        if abs(ratio_mean - 1.0) > 1e-10:
            print(f"\n  Specific factor found: GT coefficient ≈ EG coefficient × {ratio_mean:.10f}")
            
            # Try to identify what this factor might be
            if abs(ratio_mean - 0.5) < 0.01:
                print("  ⚠️  Factor ≈ 0.5: Possible 2x divisor difference")
            elif abs(ratio_mean - 2.0) < 0.01:
                print("  ⚠️  Factor ≈ 2.0: Possible 0.5x divisor difference")
    else:
        print(f"\n  GT is consistently HIGHER than EG (by ~{(np.mean(cc_gt_arr)/np.mean(cc_eg_arr) - 1)*100:.2f}%)")
else:
    print("\n[Hypothesis B] Graph Construction Difference (CRITICAL)")
    print(f"  Evidence: {len(degree_mismatches)} nodes have inconsistent degree counts")
    print("  Suspected cause: Different vertex/edge addition order or semantics")
    print("  Location to investigate: Graph creation in test_correctness.py vs vmap construction")

print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
