#include <boost/thread.hpp>
#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

#include <libmolgrid/libmolgrid.h>
namespace caffe {

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Caffe> thread_instance_;

Caffe& Caffe::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), solver_rank_(0), multiprocess_(false) { }

Caffe::~Caffe() { }

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : cublas_handle_(NULL), curand_generator_(NULL), random_generator_(),
    mode_(Caffe::CPU),cudnn_enabled_(false),
    solver_count_(1), solver_rank_(0), multiprocess_(false) {

#ifdef USE_CUDNN
  cudnn_enabled_ = true;
#endif
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  // dkoes - we will report lack of GPU elsewhere
  if (hipblasCreate(&cublas_handle_) != HIPBLAS_STATUS_SUCCESS) {
   // LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a hiprand handler.
  if (hiprandCreateGenerator(&curand_generator_, HIPRAND_RNG_PSEUDO_DEFAULT)
      != HIPRAND_STATUS_SUCCESS ||
      hiprandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != HIPRAND_STATUS_SUCCESS) {
   //LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
}

Caffe::~Caffe() {
  if (cublas_handle_) CUBLAS_CHECK(hipblasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(hiprandDestroyGenerator(curand_generator_));
  }
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  if (Get().curand_generator_) {
    CURAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    CURAND_CHECK(hiprandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        //LOG(ERROR) << "Curand not available. Skipping setting the hiprand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));

  libmolgrid::random_engine.seed(seed);
}

void Caffe::SetDevice(const int device_id) {
  int current_device;
  CUDA_CHECK(hipGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to hipSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  if(device_id >= 0) {
    CUDA_CHECK(hipSetDevice(device_id));
  }
  // Sanjay
  // else {
    //negative means pick first avail
    //cudaSetValidDevices(NULL,0);
  //}

  if (Get().cublas_handle_) CUBLAS_CHECK(hipblasDestroy(Get().cublas_handle_));
  if (Get().curand_generator_) {
    CURAND_CHECK(hiprandDestroyGenerator(Get().curand_generator_));
  }
  CUBLAS_CHECK(hipblasCreate(&Get().cublas_handle_));
  CURAND_CHECK(hiprandCreateGenerator(&Get().curand_generator_,
      HIPRAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
      cluster_seedgen()));
}

void Caffe::DeviceQuery() {
  hipDeviceProp_t prop;
  int device;
  if (hipSuccess != hipGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(hipGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  // Sanjay
  // LOG(INFO) << "Concurrent copy and execution: "
  //     << (prop.deviceOverlap ? "Yes" : "No");
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}

bool Caffe::CheckDevice(const int device_id) {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling hipFree(0).
  // hipSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, hipSetDevice() returns hipSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. hipFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = ((hipSuccess == hipSetDevice(device_id)) &&
            (hipSuccess == hipFree(0)));
  // reset any error that may have occurred.
  hipGetLastError();
  return r;
}

int Caffe::FindDevice(const int start_id) {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  int count = 0;
  CUDA_CHECK(hipGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(hipblasStatus_t error) {
  switch (error) {
  case HIPBLAS_STATUS_SUCCESS:
    return "HIPBLAS_STATUS_SUCCESS";
  case HIPBLAS_STATUS_NOT_INITIALIZED:
    return "HIPBLAS_STATUS_NOT_INITIALIZED";
  case HIPBLAS_STATUS_ALLOC_FAILED:
    return "HIPBLAS_STATUS_ALLOC_FAILED";
  case HIPBLAS_STATUS_INVALID_VALUE:
    return "HIPBLAS_STATUS_INVALID_VALUE";
  case HIPBLAS_STATUS_ARCH_MISMATCH:
    return "HIPBLAS_STATUS_ARCH_MISMATCH";
  case HIPBLAS_STATUS_MAPPING_ERROR:
    return "HIPBLAS_STATUS_MAPPING_ERROR";
  case HIPBLAS_STATUS_EXECUTION_FAILED:
    return "HIPBLAS_STATUS_EXECUTION_FAILED";
  case HIPBLAS_STATUS_INTERNAL_ERROR:
    return "HIPBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case HIPBLAS_STATUS_NOT_SUPPORTED:
    return "HIPBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case HIPBLAS_STATUS_UNKNOWN:
    return "HIPBLAS_STATUS_UNKNOWN";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(hiprandStatus_t error) {
  switch (error) {
  case HIPRAND_STATUS_SUCCESS:
    return "HIPRAND_STATUS_SUCCESS";
  case HIPRAND_STATUS_VERSION_MISMATCH:
    return "HIPRAND_STATUS_VERSION_MISMATCH";
  case HIPRAND_STATUS_NOT_INITIALIZED:
    return "HIPRAND_STATUS_NOT_INITIALIZED";
  case HIPRAND_STATUS_ALLOCATION_FAILED:
    return "HIPRAND_STATUS_ALLOCATION_FAILED";
  case HIPRAND_STATUS_TYPE_ERROR:
    return "HIPRAND_STATUS_TYPE_ERROR";
  case HIPRAND_STATUS_OUT_OF_RANGE:
    return "HIPRAND_STATUS_OUT_OF_RANGE";
  case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
  case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case HIPRAND_STATUS_LAUNCH_FAILURE:
    return "HIPRAND_STATUS_LAUNCH_FAILURE";
  case HIPRAND_STATUS_PREEXISTING_FAILURE:
    return "HIPRAND_STATUS_PREEXISTING_FAILURE";
  case HIPRAND_STATUS_INITIALIZATION_FAILED:
    return "HIPRAND_STATUS_INITIALIZATION_FAILED";
  case HIPRAND_STATUS_ARCH_MISMATCH:
    return "HIPRAND_STATUS_ARCH_MISMATCH";
  case HIPRAND_STATUS_INTERNAL_ERROR:
    return "HIPRAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown hiprand status";
}

#endif  // CPU_ONLY

}  // namespace caffe
