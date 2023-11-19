/*****************************************************************************
 *                                                                           *
 * Copyright 2016-2018 Intel Corporation.                                    *
 * Copyright 2019-2023 Alexey V. Medvedev                                    *
 *                                                                           *
 *****************************************************************************

   The 3-Clause BSD License

   Copyright (C) Intel, Inc. All rights reserved.
   Copyright (C) 2019-2023 Alexey V. Medvedev. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "async_suite.h"

class topohelper;

namespace async_suite {
    static constexpr size_t ASSUMED_CACHE_SIZE = 4 * 1024 * 1024;
    static constexpr int MAX_REQUESTS_NUM = 10;
    static constexpr size_t CALC_MATRIX_SIZE = 5;
    static constexpr bool EXTRA_BARRIER = false;
    class AsyncBenchmark : public Benchmark {
        public:
        struct result {
            bool done;
            double time;
            double overhead_comm;
	        double overhead_calc;
            int ncycles;
        };
        bool is_gpu = false;
        bool is_cuda_aware = false;
        std::map<int, result> results;
        char *host_sbuf = nullptr, *host_rbuf = nullptr;
        char *device_sbuf = nullptr, *device_rbuf = nullptr;
        int np = 0, rank = 0;
        std::shared_ptr<topohelper> topo;
        size_t allocated_size_send = 0;
        size_t allocated_size_recv = 0;
        int dtsize = 0;
        public:
        virtual void init() override;
        void alloc();
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) = 0;
        virtual void run(const scope_item &item) override; 
        virtual void finalize() override;
        virtual size_t buf_size_multiplier_send() { return 1; }
        virtual size_t buf_size_multiplier_recv() { return 1; }
        size_t get_send_bufsize_for_len(size_t len);
        size_t get_recv_bufsize_for_len(size_t len);

        void setup_the_gpu_rank();

        char *get_sbuf();
        char *get_rbuf();
        char *get_sbuf(size_t i, size_t b, size_t block = 0);
        char *get_rbuf(size_t i, size_t b, size_t block = 0);
        void sync_sbuf_with_device(size_t i, size_t off);
        void sync_rbuf_with_device(size_t i, size_t off);
            
        AsyncBenchmark() {}
        virtual ~AsyncBenchmark(); 
    };

    static void barrier(int rank, int np, const MPI_Comm &comm = MPI_COMM_WORLD) {
        if (!EXTRA_BARRIER) {
            (void)rank;
            (void)np;
            MPI_Barrier(comm);
        } else {
            // Explicit implementation of dissemenation barrier algorithm -- if we are not sure the
            // standard MPI_Barrier() is "strong" and "symmetric" enough.
            int mask = 0x1;
            int dst, src;
            int tmp = 0;
            for (; mask < np; mask <<= 1) {
                dst = (rank + mask) % np;
                src = (rank - mask + np) % np;
                MPI_Sendrecv(&tmp, 0, MPI_BYTE, dst, 1010,
                             &tmp, 0, MPI_BYTE, src, 1010,
                             comm, MPI_STATUS_IGNORE);
            }
        }
    }
}
