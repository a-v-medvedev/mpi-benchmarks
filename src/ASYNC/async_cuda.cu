#include <assert.h>
#include "async_cuda.h"


namespace sys {

namespace cuda {

void d2h_transfer(char*, char*, unsigned long) {
    assert(0 && "not implemented");
}

void h2d_transfer(char*, char*, unsigned long) {
    assert(0 && "not implemented");
}

size_t get_num_of_devices() {
    assert(0 && "not implemented");
    return 0;
}

void set_current_device(unsigned long) {
    assert(0 && "not implemented");
}

}

}

