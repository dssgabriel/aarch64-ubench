#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

extern "C" uint64_t clktest(uint64_t iterations);

extern "C" uint64_t lat_scalar_fadd(uint64_t iterations, float arr[4]);
extern "C" uint64_t lat_scalar_fmul(uint64_t iterations, float arr[4]);
extern "C" uint64_t lat_scalar_fmadd(uint64_t iterations, float arr[4]);
extern "C" uint64_t lat_neon_fadd(uint64_t iterations, float arr[4]);
extern "C" uint64_t lat_neon_fmul(uint64_t iterations, float arr[4]);
extern "C" uint64_t lat_neon_fmla(uint64_t iterations, float arr[4]);
#ifdef __ARM_FEATURE_SVE
extern "C" uint64_t lat_sve_fadd(uint64_t iterations, float arr[64]);
extern "C" uint64_t lat_sve_fmul(uint64_t iterations, float arr[64]);
extern "C" uint64_t lat_sve_fmla(uint64_t iterations, float arr[64]);
#endif

extern "C" uint64_t tp_scalar_fadd(uint64_t iterations, float arr[4]);
extern "C" uint64_t tp_scalar_fmul(uint64_t iterations, float arr[4]);
extern "C" uint64_t tp_scalar_fmadd(uint64_t iterations, float arr[4]);
extern "C" uint64_t tp_neon_fadd(uint64_t iterations, float arr[4]);
extern "C" uint64_t tp_neon_fmul(uint64_t iterations, float arr[4]);
extern "C" uint64_t tp_neon_fmla(uint64_t iterations, float arr[4]);
extern "C" uint64_t tp_neon_mix_faddfmul(uint64_t iterations, float arr[4]);
#ifdef __ARM_FEATURE_SVE
extern "C" uint64_t tp_sve_fadd(uint64_t iterations, float arr[64]);
extern "C" uint64_t tp_sve_fmul(uint64_t iterations, float arr[64]);
extern "C" uint64_t tp_sve_fmla(uint64_t iterations, float arr[64]);
extern "C" uint64_t tp_sve_mix_faddfmul(uint64_t iterations, float arr[64]);
#endif

static float fp_test_array[4] __attribute__((aligned(64))) = {0.2, 1.5, 2.7, 3.14};
static float sve_test_array[64] __attribute__((aligned(64))
) = {0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f,
     0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f,
     0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f,
     0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f, 0.2f, 1.5f, 2.7f, 3.14f};

enum InstKind {
  Scalar,
  Neon,
  Sve,
};

namespace latency {
template <InstKind Kind>
auto fadd(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return lat_scalar_fadd(iterations, fp_test_array);
  } else if constexpr (Kind == Neon) {
    return lat_neon_fadd(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return lat_sve_fadd(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}

template <InstKind Kind>
auto fmul(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return lat_scalar_fmul(iterations, fp_test_array);
  } else if constexpr (Kind == Neon) {
    return lat_neon_fmul(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return lat_sve_fmul(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}

template <InstKind Kind>
auto fmadd(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return lat_scalar_fmadd(iterations, fp_test_array);
  } else if constexpr (Kind == Neon) {
    return lat_neon_fmla(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return lat_sve_fmla(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}
}  // namespace latency

namespace throughput {

template <InstKind Kind>
auto fadd(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return tp_scalar_fadd(iterations, fp_test_array);
  } else if constexpr (Kind == Neon) {
    return tp_neon_fadd(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return tp_sve_fadd(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}

template <InstKind Kind>
auto fmul(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return tp_scalar_fmul(iterations, fp_test_array);
  } else if constexpr (Kind == Neon) {
    return tp_neon_fmul(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return tp_sve_fmul(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}

template <InstKind Kind>
auto fmadd(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return tp_scalar_fmadd(iterations, fp_test_array);
  } else if constexpr (Kind == Neon) {
    return tp_neon_fmla(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return tp_sve_fmla(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}

template <InstKind Kind>
auto mix_faddfmul(uint64_t iterations) -> uint64_t {
  if constexpr (Kind == Scalar) {
    return 0;
  } else if constexpr (Kind == Neon) {
    return tp_neon_mix_faddfmul(iterations, fp_test_array);
  } else if constexpr (Kind == Sve) {
#ifdef __ARM_FEATURE_SVE
    return tp_sve_mix_faddfmul(iterations, sve_test_array);
#else
#error "CPU does not support Arm SVE"
#endif
  } else {
    std::abort();
  }
}
}  // namespace throughput

auto measure_freq(uint64_t iterations) -> double {
  struct timespec start, stop;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  clktest(iterations);
  clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

  uint64_t time_diff_ns = 1.0e+9 * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
  double latency        = (double)time_diff_ns / (double)iterations;

  return 1 / latency;
}

auto measure_kernel(uint64_t iterations, double est_cpu_freq, uint64_t (*kernel_fn)(uint64_t)) -> double {
  struct timespec start, stop;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  kernel_fn(iterations);
  clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

  double time_diff_ns = 1.0e+9 * (double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec);
  double latency      = time_diff_ns / (double)iterations;

  return latency * est_cpu_freq;
}

int main(int argc, char *argv[]) {
  uint64_t iterations      = 100000000;
  uint64_t iterations_high = iterations * 5;

  double est_cpu_freq = measure_freq(iterations_high);
  printf("Estimated CPU frequency: %.2f GHz\n\n", est_cpu_freq);
  printf("\x1b[1m%-30s %6s %6s %6s\x1b[0m\n", "INSTRUCTION", "LAT", "TP", "1/TP");

  double thp_fadd_scalar = measure_kernel(iterations_high, est_cpu_freq, throughput::fadd<Scalar>);
  printf(
      "%-30s %6.2lf %6.2lf %6.2lf\n", "Scalar FADD", measure_kernel(iterations, est_cpu_freq, latency::fadd<Scalar>),
      thp_fadd_scalar, 1.0 / thp_fadd_scalar
  );
  double thp_fmul_scalar = measure_kernel(iterations_high, est_cpu_freq, throughput::fmul<Scalar>);
  printf(
      "%-30s %6.2lf %6.2lf %6.2lf\n", "Scalar FMUL", measure_kernel(iterations, est_cpu_freq, latency::fmul<Scalar>),
      thp_fmul_scalar, 1.0 / thp_fmul_scalar
  );
  double thp_fmadd_scalar = measure_kernel(iterations_high, est_cpu_freq, throughput::fmadd<Scalar>);
  printf(
      "%-30s %6.2lf %6.2lf %6.2lf\n", "Scalar FMADD", measure_kernel(iterations, est_cpu_freq, latency::fmadd<Scalar>),
      thp_fmadd_scalar, 1.0 / thp_fmadd_scalar
  );

  double thp_fadd_neon = measure_kernel(iterations_high, est_cpu_freq, throughput::fadd<Neon>);
  printf(
      "%-30s %6.2lf %6.2lf %6.2lf\n", "NEON FADD", measure_kernel(iterations, est_cpu_freq, latency::fadd<Neon>),
      thp_fadd_neon, 1.0 / thp_fadd_neon
  );
  double thp_fmul_neon = measure_kernel(iterations_high, est_cpu_freq, throughput::fmul<Neon>);
  printf(
      "%-30s %6.2lf %6.2lf %6.2lf\n", "NEON FMUL", measure_kernel(iterations, est_cpu_freq, latency::fmul<Neon>),
      thp_fmul_neon, 1.0 / thp_fmul_neon
  );
  double thp_fmadd_neon = measure_kernel(iterations_high, est_cpu_freq, throughput::fmadd<Neon>);
  printf(
      "%-30s %6.2lf %6.2lf %6.2lf\n", "NEON FMLA", measure_kernel(iterations, est_cpu_freq, latency::fmadd<Neon>),
      thp_fmadd_neon, 1.0 / thp_fmadd_neon
  );
  double thp_mix_faddfmul_neon = measure_kernel(iterations_high, est_cpu_freq, throughput::mix_faddfmul<Neon>);
  printf("%-30s %6s %6.2lf %6.2lf\n", "NEON 1:1 mix FADD/FMUL", "", thp_mix_faddfmul_neon, 1.0 / thp_mix_faddfmul_neon);

#ifdef __ARM_FEATURE_SVE
  double thp_fadd_sve = measure_kernel(iterations_high, est_cpu_freq, throughput::fadd<Sve>);
  printf(
      "%lu-bit %-22s %6.2lf %6.2lf %6.2lf\n", svcntw() * 32, "SVE FADD",
      measure_kernel(iterations, est_cpu_freq, latency::fadd<Sve>), thp_fadd_sve, 1.0 / thp_fadd_sve
  );
  double thp_fmul_sve = measure_kernel(iterations_high, est_cpu_freq, throughput::fmul<Sve>);
  printf(
      "%lu-bit %-22s %6.2lf %6.2lf %6.2lf\n", svcntw() * 32, "SVE FMUL",
      measure_kernel(iterations, est_cpu_freq, latency::fmul<Sve>), thp_fmul_sve, 1.0 / thp_fmul_sve
  );
  double thp_fmadd_sve = measure_kernel(iterations_high, est_cpu_freq, throughput::fmadd<Sve>);
  printf(
      "%lu-bit %-22s %6.2lf %6.2lf %6.2lf\n", svcntw() * 32, "SVE FMLA",
      measure_kernel(iterations, est_cpu_freq, latency::fmadd<Sve>), thp_fmadd_sve, 1.0 / thp_fmadd_sve
  );
  double thp_mix_faddfmul_sve = measure_kernel(iterations_high, est_cpu_freq, throughput::mix_faddfmul<Sve>);
  printf(
      "%lu-bit %-22s %6s %6.2lf %6.2lf\n", svcntw() * 32, "SVE 1:1 mix FADD/FMUL", "", thp_mix_faddfmul_sve,
      1.0 / thp_mix_faddfmul_sve
  );
#endif

  return 0;
}
