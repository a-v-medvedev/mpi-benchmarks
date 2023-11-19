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

class topohelper;

namespace async_suite {
    class AsyncBenchmark_workload : public AsyncBenchmark {
        public:
        MPI_Request *reqs = nullptr;
        int stat[MAX_REQUESTS_NUM];
        int total_tests = 0;
        int successful_tests = 0;
        int num_requests = 0;
        bool is_manual_progress = false;
        bool is_cpu_calculations = false;
        bool is_gpu_calculations = false;
        bool is_omit_calculation_loop = false;
        bool is_omit_calulation_overhead_estimation = false;

        int gpu_workload_calibration = 0;
        char *host_transfer_buf = nullptr; 
        char *device_transfer_buf = nullptr; 
        size_t gpu_workload_ncycles = 1;
        size_t gpu_workload_transfer_size = 1024;
        bool gpu_calc_cycle_active = false;
        bool gpu_calc_cycle_finish = false;
        std::thread *p_tgpucalc = nullptr;
        
        std::map<int, int> calctime_by_len;
        int cycles_per_10usec = 0; 
        int cycles_per_10usec_avg = 0, cycles_per_10usec_min = 0, cycles_per_10usec_max = 0;
        int irregularity_level = 0;
        float a[CALC_MATRIX_SIZE][CALC_MATRIX_SIZE], b[CALC_MATRIX_SIZE][CALC_MATRIX_SIZE], 
              c[CALC_MATRIX_SIZE][CALC_MATRIX_SIZE], x[CALC_MATRIX_SIZE], y[CALC_MATRIX_SIZE];
        void calc_and_progress_loop(int ncycles, int iters_till_test, double &tover_comm);
        void calc_loop(int ncycles, double &tover_comm);
        void calc_loop(int ncalccycles);
        void gpu_calc_loop();
        public:
        void calibration();
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual bool is_default() override { return false; }
        virtual void finalize() override;
        DEFINE_INHERITED(AsyncBenchmark_workload, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_calibration : public Benchmark {
        AsyncBenchmark_workload calc;
        int np = 0, rank = 0;
        public:
        virtual void init() override;
        virtual void run(const scope_item &item) override; 
        virtual void finalize() override;
        virtual bool is_default() override { return false; }
        DEFINE_INHERITED(AsyncBenchmark_calibration, BenchmarkSuite<BS_GENERIC>);
    };
}
