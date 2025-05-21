/*
ECL-MST: This code computes a minimum spanning tree (MST) or a minimum spanning forest (MSF) of an undirected graph.

Modified to read graphs from text files where each line contains two node IDs.
Also modified to output CPU and GPU runtimes, speedup, and write MST edges to a file.
*/

#include <climits>
#include <algorithm>
#include <tuple>
#include <vector>
#include <sys/time.h>
#include <cuda.h>
#include <cstdio>
#include "ECLgraph.h"

static const int Device = 0;
static const int ThreadsPerBlock = 512;

typedef unsigned long long ull;

struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return (end.tv_sec - beg.tv_sec) + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};

static inline int serial_find(const int idx, int* const parent)
{
  int curr = parent[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr != (next = parent[curr])) {
      parent[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

static inline void serial_join(const int a, const int b, int* const parent)
{
  const int arep = serial_find(a, parent);
  const int brep = serial_find(b, parent);
  if (arep > brep) {  // improves locality
    parent[brep] = arep;
  } else {
    parent[arep] = brep;
  }
}

static bool* cpuMST(const ECLgraph& g, double& cpu_runtime)
{
  bool* const inMST = new bool [g.edges];
  int* const parent = new int [g.nodes];

  CPUTimer timer;
  timer.start();

  std::fill(inMST, inMST + g.edges, false);
  for (int i = 0; i < g.nodes; i++) parent[i] = i;

  std::vector<std::tuple<int, int, int, int>> list;  // <weight, edge index, from node, to node>
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int n = g.nlist[j];
      if (n > i) {  // only one direction
        list.emplace_back(g.eweight[j], j, i, n);
      }
    }
  }
  std::sort(list.begin(), list.end());

  int count = g.nodes - 1;
  for (size_t pos = 0; pos < list.size(); pos++) {
    const int a = std::get<2>(list[pos]);
    const int b = std::get<3>(list[pos]);
    const int arep = serial_find(a, parent);
    const int brep = serial_find(b, parent);
    if (arep != brep) {
      const int j = std::get<1>(list[pos]);
      inMST[j] = true;
      serial_join(arep, brep, parent);
      count--;
      if (count == 0) break;
    }
  }

  cpu_runtime = timer.stop();
  printf("Host CPU runtime: %12.9f s\n", cpu_runtime);

  delete [] parent;
  return inMST;
}

static inline __device__ int find(int curr, const int* const __restrict__ parent)
{
  int next;
  while (curr != (next = parent[curr])) {
    curr = next;
  }
  return curr;
}

static inline __device__ void join(int arep, int brep, int* const __restrict__ parent)
{
  int mrep;
  do {
    mrep = max(arep, brep);
    arep = min(arep, brep);
  } while ((brep = atomicCAS(&parent[mrep], mrep, arep)) != mrep);
}

static __global__ void initPM(const int nodes, int* const __restrict__ parent, ull* const __restrict__ minv)
{
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (v < nodes) {
    parent[v] = v;
    minv[v] = ULONG_MAX;
  }
}

template <bool first>
static __global__ void initWL(int4* const __restrict__ wl2, int* const __restrict__ wl2size, const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist, const int* const __restrict__ eweight, ull* const __restrict__ minv, const int* const __restrict__ parent, const int threshold)
{
  const int v = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  int beg, end, arep, deg = -1;
  if (v < nodes) {
    beg = nindex[v];
    end = nindex[v + 1];
    deg = end - beg;
    arep = first ? v : find(v, parent);
    if (deg < 4) {
      for (int j = beg; j < end; j++) {
        const int n = nlist[j];
        if (n > v) {  // only in one direction
          const int wei = eweight[j];
          if (first ? (wei <= threshold) : (wei > threshold)) {
            const int brep = first ? n : find(n, parent);
            if (first || (arep != brep)) {
              const int k = atomicAdd(wl2size, 1);
              wl2[k] = int4{arep, brep, wei, j};  // <from node, to node, weight, edge index>
            }
          }
        }
      }
    }
  }
  const int WS = 32;  // warp size
  const int lane = threadIdx.x % WS;
  int bal = __ballot_sync(~0, deg >= 4);
  while (bal != 0) {
    const int who = __ffs(bal) - 1;
    bal &= bal - 1;
    const int wi = __shfl_sync(~0, v, who);
    const int wbeg = __shfl_sync(~0, beg, who);
    const int wend = __shfl_sync(~0, end, who);
    const int warep = first ? wi : __shfl_sync(~0, arep, who);
    for (int j = wbeg + lane; j < wend; j += WS) {
      const int n = nlist[j];
      if (n > wi) {  // only in one direction
        const int wei = eweight[j];
        if (first ? (wei <= threshold) : (wei > threshold)) {
          const int brep = first ? n : find(n, parent);
          if (first || (warep != brep)) {
            const int k = atomicAdd(wl2size, 1);
            wl2[k] = int4{warep, brep, wei, j};  // <from node, to node, weight, edge index>
          }
        }
      }
    }
  }
}

static __global__ void kernel1(const int4* const __restrict__ wl1, const int wl1size, int4* const __restrict__ wl2, int* const __restrict__ wl2size, const int* const __restrict__ parent, volatile ull* const __restrict__ minv)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wl1size) {
    int4 el = wl1[idx];
    const int arep = find(el.x, parent);
    const int brep = find(el.y, parent);
    if (arep != brep) {
      el.x = arep;
      el.y = brep;
      wl2[atomicAdd(wl2size, 1)] = el;
      const ull val = (((ull)el.z) << 32) | el.w;
      if (minv[arep] > val) atomicMin((ull*)&minv[arep], val);
      if (minv[brep] > val) atomicMin((ull*)&minv[brep], val);
    }
  }
}

static __global__ void kernel2(const int4* const __restrict__ wl, const int wlsize, int* const __restrict__ parent, ull* const __restrict__ minv, bool* const __restrict__ inMST)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wlsize) {
    const int4 el = wl[idx];
    const ull val = (((ull)el.z) << 32) | el.w;
    if ((val == minv[el.x]) || (val == minv[el.y])) {
      join(el.x, el.y, parent);
      inMST[el.w] = true;
    }
  }
}

static __global__ void kernel3(const int4* const __restrict__ wl, const int wlsize, volatile ull* const __restrict__ minv)
{
  const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < wlsize) {
    const int4 el = wl[idx];
    minv[el.x] = ULONG_MAX;
    minv[el.y] = ULONG_MAX;
  }
}

static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}

template <bool filter>
static bool* gpuMST(const ECLgraph& g, const int threshold, double& gpu_runtime)
{
  bool* d_inMST = NULL;
  cudaError_t err;

  // Allocate device memory
  err = cudaMalloc((void**)&d_inMST, g.edges * sizeof(bool));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_inMST: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  bool* const inMST = new bool [g.edges];

  int* d_parent = NULL;
  err = cudaMalloc((void**)&d_parent, g.nodes * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_parent: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  ull* d_minv = NULL;
  err = cudaMalloc((void**)&d_minv, g.nodes * sizeof(ull));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_minv: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  int4* d_wl1 = NULL;
  err = cudaMalloc((void**)&d_wl1, g.edges / 2 * sizeof(int4));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_wl1: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  int* d_wlsize = NULL;
  err = cudaMalloc((void**)&d_wlsize, sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_wlsize: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  int4* d_wl2 = NULL;
  err = cudaMalloc((void**)&d_wl2, g.edges / 2 * sizeof(int4));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_wl2: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  int* d_nindex = NULL;
  err = cudaMalloc((void**)&d_nindex, (g.nodes + 1) * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_nindex: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  err = cudaMemcpy(d_nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy error for d_nindex: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  int* d_nlist = NULL;
  err = cudaMalloc((void**)&d_nlist, g.edges * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_nlist: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  err = cudaMemcpy(d_nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy error for d_nlist: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  int* d_eweight = NULL;
  err = cudaMalloc((void**)&d_eweight, g.edges * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc error for d_eweight: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  err = cudaMemcpy(d_eweight, g.eweight, g.edges * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy error for d_eweight: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  CheckCuda(__LINE__);

  CPUTimer timer;
  timer.start();

  // Initialize parent and minv
  const int blocks_init = (g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock;
  initPM<<<blocks_init, ThreadsPerBlock>>>(g.nodes, d_parent, d_minv);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error for initPM: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  // Initialize inMST to false
  cudaMemset(d_inMST, 0, g.edges * sizeof(bool));
  CheckCuda(__LINE__);

  // Initialize worklist size to 0
  cudaMemset(d_wlsize, 0, sizeof(int));
  CheckCuda(__LINE__);

  // Initialize worklist
  initWL<true><<<blocks_init, ThreadsPerBlock>>>(d_wl1, d_wlsize, g.nodes, d_nindex, d_nlist, d_eweight, d_minv, d_parent, threshold);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error for initWL<true>: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  CheckCuda(__LINE__);

  // Copy worklist size from device to host
  int wlsize;
  err = cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy error for wlsize after initWL<true>: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  // Main GPU loop
  while (wlsize > 0) {
    // Reset worklist size
    cudaMemset(d_wlsize, 0, sizeof(int));
    CheckCuda(__LINE__);

    // Launch kernel1 to process current worklist and generate the next worklist
    const int blocks_k1 = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    kernel1<<<blocks_k1, ThreadsPerBlock>>>(d_wl1, wlsize, d_wl2, d_wlsize, d_parent, d_minv);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA kernel launch error for kernel1: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    CheckCuda(__LINE__);
    std::swap(d_wl1, d_wl2);

    // Copy new worklist size from device to host
    err = cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA memcpy error for wlsize after kernel1: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    if (wlsize > 0) {
      // Launch kernel2 to add edges to MST and kernel3 to reset minv
      const int blocks_k2 = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;
      kernel2<<<blocks_k2, ThreadsPerBlock>>>(d_wl1, wlsize, d_parent, d_minv, d_inMST);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error for kernel2: %s\n", cudaGetErrorString(err));
        exit(-1);
      }

      kernel3<<<blocks_k2, ThreadsPerBlock>>>(d_wl1, wlsize, d_minv);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error for kernel3: %s\n", cudaGetErrorString(err));
        exit(-1);
      }
    }
  }

  // If filtering is enabled, perform additional processing
  if (filter) {
    // Initialize worklist with filtered edges
    initWL<false><<<blocks_init, ThreadsPerBlock>>>(d_wl1, d_wlsize, g.nodes, d_nindex, d_nlist, d_eweight, d_minv, d_parent, threshold);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA kernel launch error for initWL<false>: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    CheckCuda(__LINE__);

    // Copy worklist size from device to host
    err = cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA memcpy error for wlsize after initWL<false>: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    while (wlsize > 0) {
      // Reset worklist size
      cudaMemset(d_wlsize, 0, sizeof(int));
      CheckCuda(__LINE__);

      // Launch kernel1 to process current worklist and generate the next worklist
      const int blocks_k1 = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;
      kernel1<<<blocks_k1, ThreadsPerBlock>>>(d_wl1, wlsize, d_wl2, d_wlsize, d_parent, d_minv);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error for kernel1 (filtered): %s\n", cudaGetErrorString(err));
        exit(-1);
      }
      CheckCuda(__LINE__);
      std::swap(d_wl1, d_wl2);

      // Copy new worklist size from device to host
      err = cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy error for wlsize after kernel1 (filtered): %s\n", cudaGetErrorString(err));
        exit(-1);
      }

      if (wlsize > 0) {
        // Launch kernel2 to add edges to MST and kernel3 to reset minv
        const int blocks_k2 = (wlsize + ThreadsPerBlock - 1) / ThreadsPerBlock;
        kernel2<<<blocks_k2, ThreadsPerBlock>>>(d_wl1, wlsize, d_parent, d_minv, d_inMST);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          fprintf(stderr, "CUDA kernel launch error for kernel2 (filtered): %s\n", cudaGetErrorString(err));
          exit(-1);
        }

        kernel3<<<blocks_k2, ThreadsPerBlock>>>(d_wl1, wlsize, d_minv);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          fprintf(stderr, "CUDA kernel launch error for kernel3 (filtered): %s\n", cudaGetErrorString(err));
          exit(-1);
        }
      }
    }
  }

  // Synchronize and stop timer
  cudaDeviceSynchronize();
  gpu_runtime = timer.stop();
  printf("Device GPU runtime: %12.9f s\n", gpu_runtime);

  // Copy inMST array from device to host
  err = cudaMemcpy(inMST, d_inMST, g.edges * sizeof(bool), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy error for inMST: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  // Free device memory
  cudaFree(d_inMST);
  cudaFree(d_parent);
  cudaFree(d_minv);
  cudaFree(d_wl1);
  cudaFree(d_wl2);
  cudaFree(d_wlsize);
  cudaFree(d_nindex);
  cudaFree(d_nlist);
  cudaFree(d_eweight);

  CheckCuda(__LINE__);

  return inMST;
}

static void verify(const ECLgraph& g, const bool* const cpuMSTedges, const bool* const gpuMSTedges)
{
  int bothMST = 0, neitherMST = 0, onlyCpuMST = 0, onlyGpuMST = 0;
  ull cpuMSTweight = 0, gpuMSTweight = 0;

  for (int j = 0; j < g.edges; j++) {
    const bool inCpuMST = cpuMSTedges[j];
    const bool inGpuMST = gpuMSTedges[j];
    if (inCpuMST && inGpuMST) bothMST++;
    if (!inCpuMST && !inGpuMST) neitherMST++;
    if (!inCpuMST && inGpuMST) onlyGpuMST++;
    if (inCpuMST && !inGpuMST) onlyCpuMST++;
    if (gpuMSTedges[j]) gpuMSTweight += g.eweight[j];
    if (cpuMSTedges[j]) cpuMSTweight += g.eweight[j];
  }

  if ((gpuMSTweight != cpuMSTweight) || (onlyGpuMST != 0) || (onlyCpuMST != 0)) {
    printf("ERROR: results differ!\n\n");
  } else {
    printf("All good, MST weights match.\n\n");
  }
}

// Hash function for edge sampling
static inline unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

#include <cstring>  // for std::strrchr

int main(int argc, char* argv [])
{
  printf("ECL-MST v1.0\n\n");  fflush(stdout);

  // Process command line
  if (argc != 2) {
    printf("USAGE: %s input_graph\n", argv[0]);
    exit(-1);
  }
  ECLgraph g = readECLgraph(argv[1]);
  printf("Input graph: %s\n", argv[1]);

  // Determine output file name based on input graph name
  const char* input_filename = argv[1];
  const char* base_name = std::strrchr(input_filename, '/'); // For path handling
  if (!base_name) base_name = input_filename;               // No path, just file name
  else base_name++;                                         // Skip the '/'
  
  // Construct output filename
  char output_filename[256];
  snprintf(output_filename, sizeof(output_filename), "result_%s", base_name);

  // Assign weights if needed (should not happen since we assign weights in readECLgraph)
  if (g.eweight == NULL) {
    g.eweight = new int [g.edges];
    for (int i = 0; i < g.nodes; i++) {
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int k = g.nlist[j];
        g.eweight[j] = 1 + ((i * k) % g.nodes);
        if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
      }
    }
  }

  // Get GPU info
  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    printf("ERROR: there is no CUDA capable device\n\n");
    exit(-1);
  }
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("GPU: %s with %d SMs and %d max threads per SM (%.1f MHz and %.1f MHz)\n\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);  fflush(stdout);

  // Run GPU code
  bool* gpuMSTedges;
  double gpu_runtime = 0.0;
  const int avg_deg = g.edges / g.nodes;
  if (avg_deg >= 4) {  // Use filtered code if average degree is at least 4
    // Sample 20 edges for picking threshold
    int sorted_weights[20];
    // Populate and sort edge list
    const int num_samples = std::min(g.edges, 20);
    for (int i = 0; i < num_samples; i++) {
      sorted_weights[i] = g.eweight[(hash(i) % g.edges)];
    }
    std::sort(sorted_weights, sorted_weights + num_samples);
    // Get threshold for 300% of g.nodes
    int threshold_index = (int)(3.0 * g.nodes * num_samples / g.edges);
    if (threshold_index >= 20) threshold_index = 19;
    const int threshold = sorted_weights[threshold_index];
    gpuMSTedges = gpuMST<true>(g, threshold, gpu_runtime);
  } else {  // Run non-filtered code
    gpuMSTedges = gpuMST<false>(g, INT_MAX, gpu_runtime);
  }

  // Run CPU code and compare result
  double cpu_runtime = 0.0;
  bool* cpuMSTedges = cpuMST(g, cpu_runtime);

  // Compute speedup
  double speedup = cpu_runtime / gpu_runtime;
  printf("Speedup (CPU time / GPU time): %f\n", speedup);

  // Verify results
  verify(g, cpuMSTedges, gpuMSTedges);

  // Output MST to file
  FILE* mst_file = fopen(output_filename, "w");
  if (mst_file == NULL) {
    fprintf(stderr, "ERROR: could not open output file %s\n\n", output_filename);
    exit(-1);
  }

  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      if (gpuMSTedges[j]) {
        int from_idx = i;
        int to_idx = g.nlist[j];
        // Ensure to write each edge only once
        if (to_idx > from_idx) {
          int from_id = g.index_to_id[from_idx];
          int to_id = g.index_to_id[to_idx];
          fprintf(mst_file, "(%d, %d)\n", from_id, to_id);
        }
      }
    }
  }

  fclose(mst_file);
  printf("MST has been written to %s\n", output_filename);

  // Clean up
  freeECLgraph(g);
  delete [] gpuMSTedges;
  delete [] cpuMSTedges;
  return 0;
}

