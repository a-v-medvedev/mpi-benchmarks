#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include "async_cuda.h"
#include "async_timer.h"


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

void init_contexts()
{
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_main, cudaStreamNonBlocking))
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_workload, cudaStreamNonBlocking))
    CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
}

void sync_contexts()
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

int get_current_device_hash() {
    int device_id;
    char ptr[1024];
    size_t len = 1024;
    memset(ptr, 0, len);
    CUDA_CALL(cudaGetDevice(&device_id));
    CUDA_CALL(cudaDeviceGetPCIBusId(ptr, len, device_id));
    int hash = 0;
    for (size_t i = 0; i < len / 4; i += 4) {
        if (!ptr[i*4])
            break;
        int32_t *iptr = (int32_t *)ptr + i;
        int32_t x = *iptr;
        hash ^= x;
    }
    return hash;
}

bool is_device_idle()
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
                        c[i][j] += a[i][k] * b[k][j] + N * N * ncycles;
                    }
                }
            }
        }
    }
}

void submit_workload(int ncycles, int calibration_const)
{
    constexpr int array_dim = 8;
    workload<array_dim><<<1, 1, 0, stream_workload>>>(ncycles, calibration_const);
}

int workload_calibration() {
    // Workload execution time calibration procedure. Trying to tune number of cycles
    // so that workload execution+sync time is about 100 usec
    static int cuda_workload_calibration = -1;
    if (cuda_workload_calibration != -1)
        return cuda_workload_calibration;
    cuda_workload_calibration = 1;
    const int workload_tune_maxiter = 23;
    const long target_exec_time_in_usecs = 200L;
    const long good_enough_calibration = (long)(0.95 * target_exec_time_in_usecs);
    for (int i = 0; i < workload_tune_maxiter; i++) {
        timer t;
        sys::cuda::submit_workload(1, cuda_workload_calibration);
        sys::cuda::sync_contexts();
        long execution_time_in_usecs = (long)t.stop();
        // Skip 13 first time estimations: they often include some GPU API
        // initialization time
        std::cout << ">> CUDA: execution_time_in_usecs=" << execution_time_in_usecs << " cuda_workload_calibration=" << cuda_workload_calibration << std::endl;      if (i < 13)
            continue;

        if (execution_time_in_usecs == 0)
            break;
        if (execution_time_in_usecs < good_enough_calibration) {
            auto c = int(target_exec_time_in_usecs / execution_time_in_usecs);
            if (c == 2) {
                cuda_workload_calibration *= 1.5;
            } else if (c > 1000) {
                continue;
            } else if (c > 2) {
                cuda_workload_calibration *= (c - 1);
            } else {
                auto _5percent = (int)(cuda_workload_calibration * 0.05);
                cuda_workload_calibration += (_5percent ? _5percent : 1);
            }
        } else {
            break;
        }
    }
    if (cuda_workload_calibration < 2 || cuda_workload_calibration > 1000) {
        std::cout << ">> cuda_workload_calibration=" << cuda_workload_calibration << std::endl;
        throw std::runtime_error("cuda workload calibration failed");
    }
    return cuda_workload_calibration;
    //std::cout << ">> CUDA: cuda_workload_calibration = " << cuda_workload_calibration << std::endl;
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

void d2h_transfer(char *to, char *from, size_t size, transfer_t type)
{
    CUDA_CALL(cudaMemcpyAsync(to, from, size, cudaMemcpyDeviceToHost,
                              type == transfer_t::MAIN ? stream_main : stream_workload));
    if (type == transfer_t::MAIN) {
        CUDA_CALL(cudaStreamSynchronize(stream_main))
    }
}

void h2d_transfer(char *to, char *from, size_t size, transfer_t type)
{
    CUDA_CALL(cudaMemcpyAsync(to, from, size, cudaMemcpyHostToDevice,
                              type == transfer_t::MAIN ? stream_main : stream_workload));
    if (type == transfer_t::MAIN) {
        CUDA_CALL(cudaStreamSynchronize(stream_main))
    }
}


}

}

