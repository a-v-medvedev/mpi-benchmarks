#pragma once

namespace sys {
namespace cuda {
void d2h_transfer(char *to, char *from, size_t size);
void h2d_transfer(char *to, char *from, size_t size);
size_t get_num_of_devices();
void set_current_device(size_t n);

}
}
