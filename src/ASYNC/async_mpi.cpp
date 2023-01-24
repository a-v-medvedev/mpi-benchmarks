#include <mpi.h>
#include <stdio.h>
#include <stdexcept>

#define MPI_CALL(X)          \
  {                           \
    int err = X;      \
    if (err != MPI_SUCCESS) { \
      char buf[MPI_MAX_ERROR_STRING + 30] = {0,}; \
      char mpierr[MPI_MAX_ERROR_STRING + 1] = {0,}; \
      int len = 0; \
      MPI_Error_string(err, mpierr, &len); \
      snprintf(buf, MPI_MAX_ERROR_STRING + 30, "MPI API error: %d, %s", err, mpierr); \
      throw std::runtime_error(buf);              \
    }                         \
  }


namespace sys {

namespace mpi {

void alloc(char*& ptr, size_t size) {
     MPI_CALL(MPI_Alloc_mem(size, MPI_INFO_NULL, &ptr));
}

void free(char *ptr) {
    MPI_CALL(MPI_Free_mem(ptr));
}

}

}
