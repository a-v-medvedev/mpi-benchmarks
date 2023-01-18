#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include "async_cuda.h"


#define CUDA_CALL(X) { cudaError_t err = X; if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); } cudaError_t err_last = cudaGetLastError(); if (err_last != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err_last)); } }

#define CUDADRIVER_CALL(func)                          \
  { CUresult err;                                     \
    err = func;                                       \
    if (CUDA_SUCCESS != err) {                        \
      char buf[100] = {0,}; \
      snprintf(buf, 100, "CUDA runtime API error: %d", err); \
      throw std::runtime_error(buf);              \
    }                                                 \
  }

namespace sys {

namespace cuda {

static cudaStream_t stream_main = 0, stream_workload = 0;
static cudaEvent_t event = 0;
static bool initialized = false;

void d2h_transfer(char*, char*, unsigned long) {
    assert(0 && "not implemented");
}

void h2d_transfer(char*, char*, unsigned long) {
    assert(0 && "not implemented");
}

void init_contexts()
{
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_main, cudaStreamNonBlocking))
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_workload, cudaStreamNonBlocking))
    CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
}

void sync_context()
{
    CUDA_CALL(cudaStreamSynchronize(stream_main));
    CUDA_CALL(cudaStreamSynchronize(stream_workload));
}

size_t get_num_of_devices() {
    int n;
    CUDA_CALL(cudaGetDeviceCount(&n));
    return n;
}

void set_current_device(unsigned long n) {
    assert(!initialized);
    std::cout << "GPU device set: cuda_id=" << n << std::endl;
    CUDA_CALL(cudaSetDevice(n));
    init_contexts();
    initialized = true;
    assert(0 && "not implemented");
}

void set_current(const std::string &pci_id)
{
    assert(!initialized);
    CUdevice dev;
    char devname[256];
    CUDADRIVER_CALL(cuInit(0));
    CUDADRIVER_CALL(cuDeviceGetByPCIBusId(&dev, pci_id.c_str()));
    CUDADRIVER_CALL(cuDeviceGetName(devname, 256, dev));
    std::cout << "GPU device set: pci_id=" << pci_id << ", name=" << devname << " (with hwloc)" << std::endl;
    initialized = true;
}

bool is_idle()
{
    if (event) {
        CUDA_CALL(cudaEventRecord(event, stream_workload));
        cudaError_t ret = cudaEventQuery(event);
        if (ret != cudaErrorNotReady && ret != cudaSuccess) {
            // error case: throw exception
            CUDA_CALL(ret);
        }
        if (ret == cudaErrorNotReady) {
            // stream has some load currently, not idle
            return false;
        }
    }
    return true;
}

template <int SIZE>
__global__ void workload(int ncycles, int CALIBRATION_CONST) {
    __shared__ double a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];
    while (ncycles--) {
        for (int N = 0; N < CALIBRATION_CONST; N++) {
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    for (int k = 0; k < SIZE; k++) {
                        c[i][j] += a[i][k] * b[k][j] + N * N;
                    }
                }
            }
        }
    }
}

void submit_workload(int ncycles, int calibration_const)
{
    constexpr int array_dim = 10;
    workload<array_dim><<<1, 1, 0, stream_workload>>>(ncycles, calibration_const);
}

void host_alloc(char*& ptr, size_t size) {
    CUDA_CALL(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
}

void register_mem(char* ptr, size_t size) {
    CUDA_CALL(cudaHostRegister(ptr, size, cudaHostRegisterPortable));    
}

void unregister_mem(char* ptr) {
    CUDA_CALL(cudaHostUnregister(ptr)); 
}

void device_alloc(char*& ptr, size_t size) {
    CUDA_CALL(cudaMalloc(&ptr, size));
    CUDA_CALL(cudaMemset(ptr, 0, size));
}

void host_free(char *ptr) {
    CUDA_CALL(cudaFreeHost(ptr));    
}

void device_free(char *ptr) {
    if (ptr) {
        CUDA_CALL(cudaFree(ptr));
    }
}



}

}

