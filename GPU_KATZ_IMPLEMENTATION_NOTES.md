# EGGPU Exact Katz Implementation Notes

## Overview

This contribution adds deterministic GPU Katz centrality to EasyGraph and
EGGPU, together with explicit reuse APIs for repeated queries and standardized
TSV inputs.

Katz uses incoming-edge semantics:

```text
x = alpha * A_in * x + beta
```

For an edge `u -> v`, the score of `v` accumulates the score of its incoming
neighbor `u`. The GPU implementation returns a complete score dictionary using
the graph's public node labels.

## Public APIs

- `eg.gpu_katz_centrality(G)` runs GPU Katz on a `DiGraphC` and returns a Python
  dictionary.
- `eg.prepare_gpu_katz(G)` creates an immutable `PreparedKatzContext`. The
  context owns an incoming-CSR snapshot and the GPU buffers required by Katz,
  so repeated calls to `context.run(...)` do not rebuild the graph input or GPU
  context.
- `eg.load_gpu_katz_tsv_dataset(arcs_path, node_map_path)` loads an optional
  POSIX mmap TSV dataset. `arcs.tsv` stores contiguous remapped edge IDs and
  `node_map.tsv` restores original labels and isolated nodes.

`dataset.to_digraphc()` builds a real `DiGraphC`; callers may then use the
unchanged `eg.gpu_katz_centrality(G)` API. `dataset.prepare()` instead passes
the reader-built incoming CSR to the explicit prepared context for reuse.

## Scope and Resource Semantics

The default `eg.gpu_katz_centrality(G)` behavior is unchanged. It does not
silently retain graph or GPU resources between calls. Reuse is opt-in through
`prepare_gpu_katz(G)` or `dataset.prepare()`.

This contribution does not modify EasyGraph's existing CSR cache logic.

## Validation

Validation covered Gnutella04, Gnutella08, wiki-Vote, Epinions, Slashdot, and
NotreDame. For every dataset, the `DiGraphC` public API result, the first
prepared-context result, and the second prepared-context result had identical
node-key sets and a maximum per-node absolute error of `0.0`.
