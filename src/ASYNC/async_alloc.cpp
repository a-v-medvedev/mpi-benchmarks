#include <assert.h>
#include "async_sys.h"


namespace sys {

void device_mem_alloc(char*&, unsigned long, device_alloc_t) {
    assert(0 && "not implemented");
}

void device_mem_free(char*&, device_alloc_t) {
    assert(0 && "not implemented");
}

void host_mem_alloc(char*&, unsigned long, host_alloc_t) {
    assert(0 && "not implemented");
}

void host_mem_free(char*&, host_alloc_t) {
    assert(0 && "not implemented");
}

}
