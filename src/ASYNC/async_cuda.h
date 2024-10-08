#pragma once

namespace sys {
namespace cuda {

enum transfer_t { MAIN, WORKLOAD };    

void d2h_transfer(char *to, char *from, size_t size, transfer_t type = transfer_t::MAIN);
void h2d_transfer(char *to, char *from, size_t size, transfer_t type = transfer_t::MAIN);
size_t get_num_of_devices();
void set_current_device(size_t n);
size_t get_num_of_devices();
void set_current(const std::string &pci_id);
int get_current_device_hash();

void host_alloc(char*& ptr, size_t size);
void register_mem(char* ptr, size_t size);
void unregister_mem(char* ptr);
void device_alloc(char*& ptr, size_t size);
void host_free(char *ptr);
void device_free(char *ptr);

void init_contexts();
void sync_contexts();

bool is_device_idle();
void submit_workload(int ncycles, int calibration_const);
int workload_calibration();

}
}
