/*
ECLgraph: a 32-bit binary CSR graph data structure

Modified to read graphs from text files where each line contains two node IDs.
Added mapping from node indices to original node IDs.
*/

#ifndef ECL_GRAPH
#define ECL_GRAPH

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

struct ECLgraph {
  int nodes;          // Number of nodes
  int edges;          // Number of edges
  int* nindex;        // CSR index array
  int* nlist;         // CSR neighbor list
  int* eweight;       // Edge weights
  int* index_to_id;   // Mapping from index to original node ID
};

ECLgraph readECLgraph(const char* const fname)
{
  ECLgraph g;

  FILE* f = fopen(fname, "r");
  if (f == NULL) {
    fprintf(stderr, "ERROR: could not open file %s\n\n", fname);
    exit(-1);
  }

  std::vector<std::pair<int, int>> edges;
  std::set<int> node_ids;
  int node1, node2;
  while (fscanf(f, "%d %d", &node1, &node2) == 2) {
    edges.emplace_back(node1, node2);
    node_ids.insert(node1);
    node_ids.insert(node2);
  }
  fclose(f);

  // Map node IDs to contiguous indices starting from zero
  std::map<int, int> id_to_index;
  std::vector<int> index_to_id_vector;
  index_to_id_vector.reserve(node_ids.size());

  int index = 0;
  for (int id : node_ids) {
    id_to_index[id] = index;
    index_to_id_vector.push_back(id);
    index++;
  }

  g.nodes = node_ids.size();
  g.edges = edges.size() * 2;  // Since it's undirected, store both directions

  // Build adjacency lists
  std::vector<std::vector<int>> adj(g.nodes);
  std::vector<int> adj_eweight(g.edges, 0);

  int edge_counter = 0;
  for (const auto& e : edges) {
    int u = id_to_index[e.first];
    int v = id_to_index[e.second];
    adj[u].push_back(v);
    adj[v].push_back(u);  // Undirected graph, add both directions

    // Assign weights to edges (customizable as needed)
    // Example: weight = 1 + ((u * v) % g.nodes)
    int weight = 1 + ((u * v) % g.nodes);
    if (weight < 0) weight = -weight;
    adj_eweight[edge_counter++] = weight; // u -> v
    adj_eweight[edge_counter++] = weight; // v -> u
  }

  // Build CSR representation
  g.nindex = (int*)malloc((g.nodes + 1) * sizeof(int));
  g.nlist = (int*)malloc(g.edges * sizeof(int));
  g.eweight = (int*)malloc(g.edges * sizeof(int));
  g.index_to_id = (int*)malloc(g.nodes * sizeof(int));

  if (g.nindex == NULL || g.nlist == NULL || g.eweight == NULL || g.index_to_id == NULL) {
    fprintf(stderr, "ERROR: memory allocation failed\n\n");
    exit(-1);
  }

  // Fill index_to_id mapping
  for (int i = 0; i < g.nodes; i++) {
    g.index_to_id[i] = index_to_id_vector[i];
  }

  edge_counter = 0;
  g.nindex[0] = 0;
  for (int i = 0; i < g.nodes; i++) {
    const std::vector<int>& neighbors = adj[i];
    for (int neighbor : neighbors) {
      g.nlist[edge_counter] = neighbor;
      g.eweight[edge_counter] = adj_eweight[edge_counter];
      edge_counter++;
    }
    g.nindex[i + 1] = edge_counter;
  }

  return g;
}

// Function to write graph to a binary file (unchanged)
void writeECLgraph(const ECLgraph g, const char* const fname)
{
  if ((g.nodes < 1) || (g.edges < 0)) {
    fprintf(stderr, "ERROR: node or edge count too low\n\n");
    exit(-1);
  }
  int cnt;
  FILE* f = fopen(fname, "wb");
  if (f == NULL) {
    fprintf(stderr, "ERROR: could not open file %s\n\n", fname);
    exit(-1);
  }
  cnt = fwrite(&g.nodes, sizeof(g.nodes), 1, f);
  if (cnt != 1) {
    fprintf(stderr, "ERROR: failed to write nodes\n\n");
    exit(-1);
  }
  cnt = fwrite(&g.edges, sizeof(g.edges), 1, f);
  if (cnt != 1) {
    fprintf(stderr, "ERROR: failed to write edges\n\n");
    exit(-1);
  }

  cnt = fwrite(g.nindex, sizeof(g.nindex[0]), g.nodes + 1, f);
  if (cnt != g.nodes + 1) {
    fprintf(stderr, "ERROR: failed to write neighbor index list\n\n");
    exit(-1);
  }
  cnt = fwrite(g.nlist, sizeof(g.nlist[0]), g.edges, f);
  if (cnt != g.edges) {
    fprintf(stderr, "ERROR: failed to write neighbor list\n\n");
    exit(-1);
  }
  if (g.eweight != NULL) {
    cnt = fwrite(g.eweight, sizeof(g.eweight[0]), g.edges, f);
    if (cnt != g.edges) {
      fprintf(stderr, "ERROR: failed to write edge weights\n\n");
      exit(-1);
    }
  }
  fclose(f);
}

// Function to free graph memory (unchanged)
void freeECLgraph(ECLgraph &g)
{
  if (g.nindex != NULL) free(g.nindex);
  if (g.nlist != NULL) free(g.nlist);
  if (g.eweight != NULL) free(g.eweight);
  if (g.index_to_id != NULL) free(g.index_to_id);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
  g.index_to_id = NULL;
}

#endif
