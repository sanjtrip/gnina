#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>  //  for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<>
void caffe_gpu_gemm_batch<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batch_size) {
  // Note that cublas follows fortran order (column major), and AB = (B'A')'.
  // Passing a row major matrix is equivalent to passing its transpose.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  CUBLAS_CHECK(hipblasSgemmStridedBatched(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, K*N, A, lda, M*K, &beta, C, N, M*N, batch_size));
}

template<>
void caffe_gpu_gemm_batch<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batch_size) {
  // Note that cublas follows fortran order (column major), and AB = (B'A')'.
  // Passing a row major matrix is equivalent to passing its transpose.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  CUBLAS_CHECK(hipblasDgemmStridedBatched(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, K*N, A, lda, M*K, &beta, C, N, M*N, batch_size));
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  CUBLAS_CHECK(hipblasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  CUBLAS_CHECK(hipblasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  hipblasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  CUBLAS_CHECK(hipblasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  hipblasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  CUBLAS_CHECK(hipblasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(hipblasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(hipblasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(hipMemcpy(Y, X, N, hipMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(hipblasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(hipblasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           hipStream_t str) {
  hipStream_t initial_stream;
  CUBLAS_CHECK(hipblasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(hipblasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(hipblasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(hipblasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            hipStream_t str) {
  hipStream_t initial_stream;
  CUBLAS_CHECK(hipblasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(hipblasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(hipblasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(hipblasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(hipblasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(hipblasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(hipblasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(hipblasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(hipblasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(hipblasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(hipblasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(hipblasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(hipMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(set_kernel<Dtype>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_scalar_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_scalar_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(sub_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(sub_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(div_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(div_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(abs_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(abs_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(exp_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(exp_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(log_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(log_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(powx_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(powx_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(sqrt_kernel<float>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(sqrt_kernel<double>), CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, 0, 
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(hiprandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(hiprandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(hiprandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      hiprandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      hiprandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
