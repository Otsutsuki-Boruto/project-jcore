#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "adaptive_tuner.h"
#include "jcore_isa_dispatch.h"
#include "k_kernel_dispatch.h"

/* External symbol from your dispatch system */
extern "C" const char *k_dispatch_get_last_selected_kernel(void);

/* Utility: time in microseconds */
static double now_usec()
{
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* Statistics struct */
struct Stats
{
  size_t runs{};
  double mean{};
  double median{};
  double stddev{};
  double min{};
  double max{};
  double gflops{}; // Added GFLOPS
};

/* Compute summary stats */
Stats summarize(const std::vector<double> &samples, size_t M, size_t N, size_t K)
{
  Stats s{};
  if (samples.empty())
    return s;
  s.runs = samples.size();
  s.min = s.max = samples[0];
  double sum = 0.0;
  for (double v : samples)
  {
    sum += v;
    s.min = std::min(s.min, v);
    s.max = std::max(s.max, v);
  }
  s.mean = sum / samples.size();

  std::vector<double> tmp = samples;
  std::sort(tmp.begin(), tmp.end());
  s.median = (tmp.size() % 2 == 1)
                 ? tmp[tmp.size() / 2]
                 : 0.5 * (tmp[tmp.size() / 2 - 1] + tmp[tmp.size() / 2]);
  double var = 0.0;
  for (double v : samples)
    var += (v - s.mean) * (v - s.mean);
  s.stddev = std::sqrt(var / samples.size());

  // Calculate GFLOPS: 2*M*N*K operations, mean time in microseconds
  double ops = 2.0 * M * N * K;
  double mean_sec = s.mean / 1e6;
  s.gflops = (ops / mean_sec) / 1e9;

  return s;
}

/* Benchmark derived dispatch */
Stats bench_derived_kernel(const float *A, const float *B, float *C,
                           size_t M, size_t N, size_t K, size_t iterations)
{
  std::vector<double> samples(iterations);
  for (size_t it = 0; it < iterations; ++it)
  {
    double t0 = now_usec();
    k_dispatch_matmul(A, B, C, M, N, K);  // adaptive pick occurs here
    double t1 = now_usec();
    samples[it] = t1 - t0;
  }
  return summarize(samples, M, N, K);
}

/* Print detailed report */
void print_report(const std::string &kernel, const Stats &s, size_t M, size_t N, size_t K)
{
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n[Matrix Size: " << M << "x" << N << "x" << K << "]\n";
  std::cout << "Selected Kernel: " << (kernel.empty() ? "<none>" : kernel) << "\n";
  std::cout << "---------------------------------------------------------------------------------\n";
  std::cout << std::left << std::setw(25) << "Kernel Dispatch Type"
            << std::setw(10) << "Runs"
            << std::setw(12) << "Mean(us)"
            << std::setw(12) << "Median"
            << std::setw(12) << "Min"
            << std::setw(12) << "Max"
            << std::setw(12) << "StdDev"
            << std::setw(12) << "GFLOPS"
            << "\n";
  std::cout << "---------------------------------------------------------------------------------\n";
  std::cout << std::left << std::setw(25) << "DerivedDispatch"
            << std::setw(10) << s.runs
            << std::setw(12) << s.mean
            << std::setw(12) << s.median
            << std::setw(12) << s.min
            << std::setw(12) << s.max
            << std::setw(12) << s.stddev
            << std::setw(12) << s.gflops
            << "\n";
}

/* Validation test - compute naive matmul and compare */
bool validate_matmul(size_t M, size_t N, size_t K)
{
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_dispatch(M * N, 0.0f);
  std::vector<float> C_expected(M * N, 0.0f);

  // Fill with simple values
  for (size_t i = 0; i < A.size(); ++i)
    A[i] = float((i + 1) % 10) * 0.1f;
  for (size_t i = 0; i < B.size(); ++i)
    B[i] = float((i + 3) % 10) * 0.1f;

  // Compute via dispatch
  k_dispatch_matmul(A.data(), B.data(), C_dispatch.data(), M, N, K);

  // Compute expected (naive)
  for (size_t i = 0; i < M; ++i)
  {
    for (size_t j = 0; j < N; ++j)
    {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k)
      {
        sum += A[i * K + k] * B[k * N + j];
      }
      C_expected[i * N + j] = sum;
    }
  }

  // Compare
  bool passed = true;
  float max_error = 0.0f;
  for (size_t i = 0; i < M * N; ++i)
  {
    float error = std::abs(C_dispatch[i] - C_expected[i]);
    max_error = std::max(max_error, error);
    if (error > 0.01f)
    {
      if (passed)
      {
        std::cerr << "\n[VALIDATION FAILED] Size " << M << "x" << N << "x" << K << "\n";
        std::cerr << "First error at index " << i << ": "
                  << "expected=" << C_expected[i]
                  << ", got=" << C_dispatch[i] << "\n";
      }
      passed = false;
    }
  }

  if (passed)
  {
    std::cout << "[VALIDATION PASSED] Size " << M << "x" << N << "x" << K
              << " | max_error=" << std::scientific << max_error << std::fixed << "\n";
  }
  else
  {
    std::cerr << "Selected kernel: " << k_dispatch_get_last_selected_kernel() << "\n";
  }

  return passed;
}

/* Entry point */
int main()
{
  std::cout << "=== Kernel Dispatch Table - Detailed Regressor Report with GFLOPS ===\n";

  if (k_dispatch_init() != JCORE_OK)
  {
    std::cerr << "Derived kernel dispatch init failed\n";
    return 1;
  }

  std::cout << "\n=== Running Validation Tests ===\n";
  validate_matmul(32, 48, 64);
  validate_matmul(64, 64, 64);
  validate_matmul(128, 128, 128);
  std::cout << "=================================\n";

  const size_t ITER = 50;

  struct TestCase
  {
    size_t M, N, K;
  };

  std::vector<TestCase> tests = {
    {32, 32, 32},
      {64, 64, 64},
      {128, 128, 128},
      {256, 256, 256},
      {512, 512, 512},
      {1024, 1024, 1024},
      {2048, 2048, 2048},
      {4096, 4096, 4096}
  };

  for (auto &t : tests)
  {
    std::vector<float> A(t.M * t.K);
    std::vector<float> B(t.K * t.N);
    std::vector<float> C(t.M * t.N, 0.0f);

    for (size_t i = 0; i < A.size(); ++i)
      A[i] = float((i + 1) & 255) * 0.001f;
    for (size_t i = 0; i < B.size(); ++i)
      B[i] = float((i + 7) & 255) * 0.001f;

    std::cout << "\nBenchmarking Derived Dispatch for size " << t.M << "x" << t.N << "x" << t.K << "\n";

    Stats s = bench_derived_kernel(A.data(), B.data(), C.data(), t.M, t.N, t.K, ITER);
    const char *selected = k_dispatch_get_last_selected_kernel();
    print_report(selected ? selected : "<unknown>", s, t.M, t.N, t.K);

  }

  // Functional validation
  std::vector<float> As(8 * 8), Bs(8 * 8), Cs(8 * 8, 0.0f);
  for (size_t i = 0; i < As.size(); ++i)
    As[i] = float(i + 1);
  for (size_t i = 0; i < Bs.size(); ++i)
    Bs[i] = float(i + 2);

  int rc = k_dispatch_matmul(As.data(), Bs.data(), Cs.data(), 8, 8, 8);
  if (rc == JCORE_OK)
    std::cout << "\nFunctional dispatch OK | sample C[0]=" << Cs[0]
              << " C[last]=" << Cs.back() << "\n";
  else
    std::cerr << "Functional dispatch failed\n";

  k_dispatch_shutdown();
  std::cout << "\n=== End of Detailed Report ===\n";
  return 0;
}