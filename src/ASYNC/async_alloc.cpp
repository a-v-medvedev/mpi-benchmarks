#include <assert.h>
#include "async_sys.h"


namespace sys {

void device_mem_alloc(char*& ptr, unsigned long size, device_alloc_t da) {
#ifdef WITH_CUDA
    (void)da;
    sys::cuda::device_alloc(ptr, size);
#else
    (void )size;
    (void)da;
    ptr = 0;    
#endif    
}

void device_mem_free(char*& ptr, device_alloc_t da) {
#ifdef WITH_CUDA
    (void)da;
    sys::cuda::device_free(ptr);
#else
    (void)da;
    ptr = 0;    
#endif    
}

void host_mem_alloc(char*& ptr, unsigned long size, host_alloc_t ha) {
#ifdef WITH_CUDA
    //assert(0 && "not implemented");
    if (ha == host_alloc_t::HA_CUDA) {
        sys::cuda::host_alloc(ptr, size);
    } else if (ha == host_alloc_t::HA_MPI) {
        sys::mpi::alloc(ptr, size);
        sys::cuda::register_mem(ptr, size);
    } else {
        ptr = (char *)calloc(size, 1);
    }
#else
    assert(ha == host_alloc_t::HA_STDC);
    ptr = (char *)calloc(size, 1);
#endif    
}

void host_mem_free(char*& ptr, host_alloc_t ha) {
#ifdef WITH_CUDA
//    assert(0 && "not implemented");
    if (ha == host_alloc_t::HA_CUDA) {
        sys::cuda::host_free(ptr);
    } else if (ha == host_alloc_t::HA_MPI) {
        sys::cuda::unregister_mem(ptr);
        sys::mpi::free(ptr);
    } else {
        free(ptr);;
    }
#else
    assert(ha == host_alloc_t::HA_STDC);
    if (ptr != 0)
        free(ptr);
    ptr = 0;
#endif    
}

}
