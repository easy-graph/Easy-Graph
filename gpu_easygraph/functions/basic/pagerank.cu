#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#define TILE_WIDTH 32

using Clock = std::chrono::steady_clock;

double TimeCost(const Clock::time_point& start_time,
                const Clock::time_point& end_time) {
  std::chrono::microseconds time_span =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time);

  return time_span.count() * 1e-3;
}

void DebugMatrix(const float* matrix, size_t row, size_t col) {
  for (size_t i = 0; i < row; ++i) {
    std::ostringstream oss;
    for (size_t j = 0; j < col; j++) {
      oss << matrix[i * col + j] << " ";
    }
    std::cout << oss.str() << std::endl;
  }
}

// CUDA kernel for matrix multiplication
__global__ void MatrixMulKernel(float* A, float* B, float* C, int n) {
  // Calculate the row and column index of the element
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0;
  if (row < n && col < n) {
    // Perform the multiplication and sum
    for (int i = 0; i < n; ++i) {
      sum += A[row * n + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}

__global__ void FixMatrix(float* M, int n, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    M[row * n + col] = beta * M[row * n + col] + (1 - beta) * 1.0f / n;
  }
}

__global__ void CalculatePR(float* M, float* col_vec, float* res, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0;

    for (int i = 0; i < n; i++) {
      sum += M[row * n + i] * col_vec[i];
    }

    res[row] = sum;
  }
}

void MatrixMulCUDA(const float* h_A, const float* origin_vec, float* res_vec,
                   int n, int loops, float beta) {
  if (h_A == nullptr) {
    std::cout << "matrix h_A is nullptr!" << std::endl;

    return;
  }

  if (origin_vec == nullptr) {
    std::cout << "matrix origin_vec is nullptr!" << std::endl;

    return;
  }

  if (res_vec == nullptr) {
    std::cout << "matrix res_vec is nullptr!" << std::endl;

    return;
  }

  if (n <= 1) {
    std::cout << "matrix col  value is invalid! Net analysis failed! n:" << n
              << std::endl;

    return;
  }

  if (loops < 0) {
    std::cout << "loops value is invalid! Net analysis failed! loops:" << loops
              << std::endl;

    return;
  }

  if (beta > 1.0 || beta < 0.0) {
    std::cout << "beta value is invalid! Net analysis failed! beta:" << beta
              << std::endl;

    return;
  }

  // Allocate device memory
  float *d_A, *tmp_matrix, *final_matrix, *d_origin_vec, *d_res_vec;
  size_t matrix_size = n * n * sizeof(float);
  size_t vec_size = n * sizeof(float);

  float* fix_matrix_A = (float*)malloc(matrix_size);
  float* h_origin_vec = (float*)malloc(vec_size);

  cudaMalloc((void**)&d_A, matrix_size);
  cudaMalloc((void**)&tmp_matrix, matrix_size);
  cudaMalloc((void**)&final_matrix, matrix_size);
  cudaMalloc((void**)&d_origin_vec, vec_size);
  cudaMalloc((void**)&d_res_vec, vec_size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_origin_vec, origin_vec, vec_size, cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + TILE_WIDTH - 1) / TILE_WIDTH,
               (n + TILE_WIDTH - 1) / TILE_WIDTH);
  FixMatrix<<<gridDim, blockDim>>>(d_A, n, beta);
  cudaDeviceSynchronize();

  cudaMemcpy(tmp_matrix, d_A, matrix_size, cudaMemcpyDeviceToDevice);
  cudaMemcpy(final_matrix, d_A, matrix_size, cudaMemcpyDeviceToDevice);

  Clock::time_point t3 = Clock::now();

  // Launch the matrix multiplication kernel
  for (int i = 0; i < loops; ++i) {
    MatrixMulKernel<<<gridDim, blockDim>>>(d_A, tmp_matrix, final_matrix, n);
    cudaDeviceSynchronize();
    cudaMemcpy(tmp_matrix, final_matrix, matrix_size, cudaMemcpyDeviceToDevice);
  }

  CalculatePR<<<gridDim, blockDim>>>(final_matrix, d_origin_vec, d_res_vec, n);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(res_vec, d_res_vec, vec_size, cudaMemcpyDeviceToHost);
  Clock::time_point t4 = Clock::now();
  std::cout << "GPU Res:" << std::endl;
  DebugMatrix(res_vec, n, 1);
  double gpu_cost_time_ms = TimeCost(t3, t4);
  std::cout << "GPU_time_cost_ms:" << gpu_cost_time_ms << std::endl;

  // Free device memory
  cudaFree(d_A);
  cudaFree(tmp_matrix);
  cudaFree(final_matrix);
  cudaFree(d_origin_vec);
  cudaFree(d_res_vec);
}

// matrix_a m*l  matrix_b l*n
bool MatrixMulCPU(const float* matrix_a, const float* matrix_b, const int m,
                  const int l, const int n, float* matrix_c) {
  if (matrix_a == nullptr) {
    std::cout << "matrix_a is nullptr!" << std::endl;

    return false;
  }

  if (matrix_b == nullptr) {
    std::cout << "matrix_b is nullptr!" << std::endl;

    return false;
  }

  if (m <= 0 || l <= 0 || n <= 0) {
    std::cout << "matrix size is invalid! m:" << m << " l:" << n << " n:" << n
              << std::endl;

    return false;
  }

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0;
      for (size_t k = 0; k < l; ++k) {
        sum += matrix_a[i * l + k] * matrix_b[k * n + j];
      }
      matrix_c[i * n + j] = sum;
    }
  }

  return true;
}

bool NetAnalysisAlgorithm(const float* matrix_m, const float* original_vec,
                          const int n, const int loops, const float beta,
                          float* res_vec) {
  Clock::time_point t1 = Clock::now();
  if (matrix_m == nullptr) {
    std::cout << "matrix_m is nullptr! Net analysis failed!\n";

    return false;
  }

  if (original_vec == nullptr) {
    std::cout << "original_vec is nullptr! Net analysis failed!\n";

    return false;
  }

  if (beta > 1.0 || beta < 0.0) {
    std::cout << "beta value is invalid! Net analysis failed! beta:" << beta
              << std::endl;

    return false;
  }

  if (n <= 1) {
    std::cout << "matrix col  value is invalid! Net analysis failed! n:" << n
              << std::endl;

    return false;
  }

  if (loops < 0) {
    std::cout << "loops value is invalid! Net analysis failed! loops:" << loops
              << std::endl;

    return false;
  }

  if (res_vec == nullptr) {
    std::cout << "res_vec is nullptr! Net analysis failed!\n";

    return false;
  }

  float* fix_matrix_m = (float*)malloc(n * n * sizeof(float));
  for (int i = 0; i < n * n; ++i) {
    fix_matrix_m[i] = beta * matrix_m[i] + (1.0f - beta) * 1.0f / n;
  }

  float* tmp_matrix = (float*)malloc(n * n * sizeof(float));
  memcpy(tmp_matrix, fix_matrix_m, n * n * sizeof(float));
  float* final_matrix = (float*)malloc(n * n * sizeof(float));
  memcpy(final_matrix, fix_matrix_m, n * n * sizeof(float));

  bool get_res = false;
  for (int i = 0; i < loops; ++i) {
    get_res = MatrixMulCPU(fix_matrix_m, tmp_matrix, n, n, n, final_matrix);
    if (get_res) {
      memcpy(tmp_matrix, final_matrix, n * n * sizeof(float));
    } else {
      std::cout << "loops index:" << i << " failed!\n";
      free(fix_matrix_m);
      free(final_matrix);
      free(tmp_matrix);

      return false;
    }
  }

  get_res = MatrixMulCPU(final_matrix, original_vec, n, n, 1, res_vec);
  Clock::time_point t2 = Clock::now();
  std::cout << "CPU Res:" << std::endl;
  DebugMatrix(res_vec, n, 1);
  double cpu_cost_time_ms = TimeCost(t1, t2);
  std::cout << "cpu_time_cost_ms:" << cpu_cost_time_ms << std::endl;

  free(fix_matrix_m);
  free(final_matrix);
  free(tmp_matrix);

  return get_res;
}

int main() {
  const int n = 3;
  const int loops = 1000;
  const float beta = 0.85;
  size_t original_vec_size = n * sizeof(float);

  // Allocate host memory
  float initValue[] = {1, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0};
  float* h_A = initValue;
  float* h_original_vec = (float*)malloc(original_vec_size);
  float* res_vec_GPU = (float*)malloc(original_vec_size);
  float* res_vec_CPU = (float*)malloc(original_vec_size);

  for (int j = 0; j < n; ++j) {
    h_original_vec[j] = 1.0f / n;
  }

  // Use CPU to calculate
  NetAnalysisAlgorithm(h_A, h_original_vec, n, loops, beta, res_vec_CPU);

  // Perform matrix multiplication on the GPU
  MatrixMulCUDA(h_A, h_original_vec, res_vec_GPU, n, loops, beta);

  // Free host memory
  free(h_original_vec);
  free(res_vec_GPU);
  free(res_vec_CPU);

  return 0;
}
