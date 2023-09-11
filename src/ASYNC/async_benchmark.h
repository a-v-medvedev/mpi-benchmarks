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

#include "async_suite.h"

class topohelper;

namespace async_suite {
    static constexpr size_t ASSUMED_CACHE_SIZE = 4 * 1024 * 1024;
    static constexpr int MAX_REQUESTS_NUM = 10;
    static constexpr size_t CALC_MATRIX_SIZE = 5;
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
        size_t allocated_size = 0;
        int dtsize = 0;
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) = 0;
        virtual void run(const scope_item &item) override; 
        virtual void finalize() override;
        virtual size_t buf_size_multiplier() { return 1; }

        void setup_the_gpu_rank();

        char *get_sbuf();
        char *get_rbuf();
        void update_sbuf(size_t off, size_t size);
        void update_rbuf(size_t off, size_t size);
            
        AsyncBenchmark() {}
        virtual ~AsyncBenchmark(); 
    };

    class AsyncBenchmark_calc : public AsyncBenchmark {
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
        DEFINE_INHERITED(AsyncBenchmark_calc, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_calibration : public Benchmark {
        AsyncBenchmark_calc calc;
        int np = 0, rank = 0;
        public:
        virtual void init() override;
        virtual void run(const scope_item &item) override; 
        virtual void finalize() override;
        virtual bool is_default() override { return false; }
        DEFINE_INHERITED(AsyncBenchmark_calibration, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_pt2pt : public AsyncBenchmark {
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_pt2pt, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_ipt2pt : public AsyncBenchmark {
        public:
        AsyncBenchmark_calc calc;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_ipt2pt, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_allreduce : public AsyncBenchmark {
        public:
        MPI_Comm coll_comm;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_allreduce, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_iallreduce : public AsyncBenchmark {
        public:
        MPI_Comm coll_comm;
        AsyncBenchmark_calc calc;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_iallreduce, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_na2a : public AsyncBenchmark {
        public:
        MPI_Comm graph_comm;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual size_t buf_size_multiplier() override { return 2; }
        DEFINE_INHERITED(AsyncBenchmark_na2a, BenchmarkSuite<BS_GENERIC>);

    };

    class AsyncBenchmark_ina2a : public AsyncBenchmark {
        public:
        AsyncBenchmark_calc calc;
        MPI_Comm graph_comm;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual size_t buf_size_multiplier() override { return 2; }
        DEFINE_INHERITED(AsyncBenchmark_ina2a, BenchmarkSuite<BS_GENERIC>);
    };
 
    class AsyncBenchmark_rma_pt2pt : public AsyncBenchmark {
        public:
        MPI_Win win;
        char *win_buf = nullptr;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual ~AsyncBenchmark_rma_pt2pt();
        DEFINE_INHERITED(AsyncBenchmark_rma_pt2pt, BenchmarkSuite<BS_GENERIC>);

    };

    class AsyncBenchmark_rma_ipt2pt : public AsyncBenchmark {
        public:
        AsyncBenchmark_calc calc;
        MPI_Win win;
        char *win_buf = nullptr;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual ~AsyncBenchmark_rma_ipt2pt();
        DEFINE_INHERITED(AsyncBenchmark_rma_ipt2pt, BenchmarkSuite<BS_GENERIC>);
    };
}
