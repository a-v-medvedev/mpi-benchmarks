#pragma once

namespace sys {
enum host_alloc_t { HA_STDC, HA_MPI, HA_CUDA };
enum device_alloc_t { DA_CUDA };

void device_mem_alloc(char *&device_buf, size_t size_to_alloc, device_alloc_t da_type);
void host_mem_alloc(char *&host_buf, size_t size_to_alloc, host_alloc_t ha_type);
void device_mem_free(char *&device_buf, device_alloc_t da_type);
void host_mem_free(char *&host_buf, host_alloc_t ha_type);

namespace mpi {
void alloc(char*&ptr, size_t size);
void free(char *ptr);
}

}
