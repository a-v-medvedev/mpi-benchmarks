#pragma once

namespace sys {
namespace cuda {
void d2h_transfer(char *to, char *from, size_t size);
void h2d_transfer(char *to, char *from, size_t size);
size_t get_num_of_devices();
void set_current_device(size_t n);

void host_alloc(char*& ptr, size_t size);
void register_mem(char* ptr, size_t size);
void unregister_mem(char* ptr);
void device_alloc(char*& ptr, size_t size);
void host_free(char *ptr);
void device_free(char *ptr);
}
}
