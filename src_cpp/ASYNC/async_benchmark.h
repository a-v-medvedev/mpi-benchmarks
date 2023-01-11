/*****************************************************************************
 *                                                                           *
 * Copyright 2016-2018 Intel Corporation.                                    *
 *                                                                           *
 *****************************************************************************

This code is covered by the Community Source License (CPL), version
1.0 as published by IBM and reproduced in the file "license.txt" in the
"license" subdirectory. Redistribution in source and binary form, with
or without modification, is permitted ONLY within the regulations
contained in above mentioned license.

Use of the name and trademark "Intel(R) MPI Benchmarks" is allowed ONLY
within the regulations of the "License for Use of "Intel(R) MPI
Benchmarks" Name and Trademark" as reproduced in the file
"use-of-trademark-license.txt" in the "license" subdirectory.

THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT
LIMITATION, ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. Each Recipient is
solely responsible for determining the appropriateness of using and
distributing the Program and assumes all risks associated with its
exercise of rights under this Agreement, including but not limited to
the risks and costs of program errors, compliance with applicable
laws, damage to or loss of data, programs or equipment, and
unavailability or interruption of operations.

EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, NEITHER RECIPIENT NOR
ANY CONTRIBUTORS SHALL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING
WITHOUT LIMITATION LOST PROFITS), HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OR
DISTRIBUTION OF THE PROGRAM OR THE EXERCISE OF ANY RIGHTS GRANTED
HEREUNDER, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

EXPORT LAWS: THIS LICENSE ADDS NO RESTRICTIONS TO THE EXPORT LAWS OF
YOUR JURISDICTION. It is licensee's responsibility to comply with any
export regulations applicable in licensee's jurisdiction. Under
CURRENT U.S. export regulations this software is eligible for export
from the U.S. and can be downloaded by or otherwise exported or
reexported worldwide EXCEPT to U.S. embargoed destinations which
include Cuba, Iraq, Libya, North Korea, Iran, Syria, Sudan,
Afghanistan and any other country to which the U.S. has embargoed
goods and services.

 ***************************************************************************
*/

#include "async_suite.h"

class topohelper;

namespace async_suite {
    static constexpr size_t ASSUMED_CACHE_SIZE = 4 * 1024 * 1024;
    static constexpr int MAX_REQUESTS_NUM = 10;
    static constexpr size_t CALC_MATRIX_SIZE = 7;
    class AsyncBenchmark : public Benchmark {
        public:
//        const size_t ASSUMED_CACHE_SIZE = 4 * 1024 * 1024;
        struct result {
            bool done;
            double time;
            double overhead_comm;
	        double overhead_calc;
            int ncycles;
        };
        std::map<int, result> results;
        char *sbuf, *rbuf;
        int np, rank;
        std::shared_ptr<topohelper> topo;
        size_t allocated_size;
        int dtsize;
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) = 0;
        virtual void run(const scope_item &item) override; 
        virtual void finalize() override;
        virtual size_t buf_size_multiplier() { return 1; }
        AsyncBenchmark() : sbuf(nullptr), rbuf(nullptr), np(0), rank(0), allocated_size(0), dtsize(0) {}
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
        int irregularity_level = 0;
        std::map<int, int> calctime_by_len;
        //static const int SIZE = 7;
        int cycles_per_10usec_avg = 0, cycles_per_10usec_min = 0, cycles_per_10usec_max = 0;
        float a[CALC_MATRIX_SIZE][CALC_MATRIX_SIZE], b[CALC_MATRIX_SIZE][CALC_MATRIX_SIZE], 
              c[CALC_MATRIX_SIZE][CALC_MATRIX_SIZE], x[CALC_MATRIX_SIZE], y[CALC_MATRIX_SIZE];
        void calc_and_progress_cycle(int ncycles, int iters_till_test, double &tover_comm);
        void calc_cycle(int ncycles, double &tover_comm);
        public:
        void calibration();
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual bool is_default() override { return false; }
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
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        DEFINE_INHERITED(AsyncBenchmark_allreduce, BenchmarkSuite<BS_GENERIC>);
    };

    class AsyncBenchmark_iallreduce : public AsyncBenchmark {
        public:
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
        char *win_buf;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual ~AsyncBenchmark_rma_pt2pt();
        DEFINE_INHERITED(AsyncBenchmark_rma_pt2pt, BenchmarkSuite<BS_GENERIC>);

    };

    class AsyncBenchmark_rma_ipt2pt : public AsyncBenchmark {
        public:
        AsyncBenchmark_calc calc;
        MPI_Win win;
        char *win_buf;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, double &time, double &tover_comm, double &tover_calc) override;
        virtual ~AsyncBenchmark_rma_ipt2pt();
        DEFINE_INHERITED(AsyncBenchmark_rma_ipt2pt, BenchmarkSuite<BS_GENERIC>);
    };
}
