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
#include "async_benchmark.h"

namespace async_suite {

#ifndef ASYNC_EXTRA_BARRIER
#define ASYNC_EXTRA_BARRIER 0
#endif

    static void barrier(int rank, int np, const MPI_Comm &comm = MPI_COMM_WORLD) {
#if !ASYNC_EXTRA_BARRIER
        (void)rank;
        (void)np;
        MPI_Barrier(comm);
#else
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
#endif
    }
    
    class AsyncBenchmark_allreduce_base : public AsyncBenchmark {
        public:
        MPI_Comm coll_comm;
        virtual void init() override;
    };

    class AsyncBenchmark_allreduce : public AsyncBenchmark_allreduce_base {
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_allreduce, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_iallreduce : public AsyncBenchmark_allreduce_base {
        public:
        AsyncBenchmark_calc calc;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_iallreduce, BenchmarkSuite<BS_GENERIC>);
    };
}
