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

#include <thread>
#include "async_pt2pt.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_topology.h"
#include "async_cuda.h"

namespace async_suite {

    void AsyncBenchmark_pt2pt_base::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("pt2pt"), np, rank);
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_pt2pt::init() {
        AsyncBenchmark_pt2pt_base::init();
    }

    void AsyncBenchmark_ipt2pt::init() {
        AsyncBenchmark_pt2pt_base::init();
        calc.init();
    }

    bool AsyncBenchmark_pt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                         double &time, double &tover_comm, double &tover_calc) {
        tover_comm = 0;
        tover_calc = 0;
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0;
        const int tag = 1;
        auto comm_actions = topo->comm_actions();
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i == nwarmup) t1 = MPI_Wtime();
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int rank = comm_actions[commstage].rank;
                if (comm_actions[commstage].action == action_t::SEND) {
                    sync_sbuf_with_device(i, b);
                    MPI_Send(get_sbuf(i, b), count, datatype, rank, tag, MPI_COMM_WORLD);
                } else if (comm_actions[commstage].action == action_t::RECV) {
                    MPI_Recv(get_rbuf(i, b), count, datatype, rank, MPI_ANY_TAG, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    sync_rbuf_with_device(i, b);
                }
            }
        }
        t2 = MPI_Wtime();
        time = (t2 - t1) / ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0.0, 0.0, ncycles };
        return true;
    }

    bool AsyncBenchmark_ipt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                          double &time, double &tover_comm, double &tover_calc) {
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0, time_calc = 0, total_ctime = 0, total_tover_comm = 0, total_calc_slowdown_ratio = 0, 
                                              local_ctime = 0, local_tover_comm = 0, local_calc_slowdown_ratio = 0;
        const int tag = 1;
        auto comm_actions = topo->comm_actions();
        calc.num_requests = topo->get_num_actions();
        if (calc.num_requests > MAX_REQUESTS_NUM) {
            throw std::runtime_error("AsyncBenchmark_ipt2pt: MAX_REQUESTS_NUM is too little for the topology");         
        }
        MPI_Request *requests;
        requests = (MPI_Request *)calloc(sizeof(MPI_Request), calc.num_requests);
        calc.reqs = requests;
        for (int i = 0; i < ncycles + nwarmup; i++) {
            int nr = 0;
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int rank = comm_actions[commstage].rank;
                if (i == nwarmup) t1 = MPI_Wtime();
                if (comm_actions[commstage].action == action_t::SEND) {
                    sync_sbuf_with_device(i, b);
                    MPI_Isend(get_sbuf(i, b), count, datatype, rank, tag, MPI_COMM_WORLD, 
                              &requests[nr++]);
                } else if (comm_actions[commstage].action == action_t::RECV) {
                    MPI_Irecv(get_rbuf(i, b), count, datatype, rank, MPI_ANY_TAG, MPI_COMM_WORLD, 
                              &requests[nr++]);
                }
            }
            calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_calc_slowdown_ratio);
            
            if (i >= nwarmup) {
                total_ctime += local_ctime;
                total_tover_comm += local_tover_comm;
                total_calc_slowdown_ratio += local_calc_slowdown_ratio;
            }
            MPI_Waitall(calc.num_requests, requests, MPI_STATUSES_IGNORE);
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                if (comm_actions[commstage].action == action_t::RECV) {
                    sync_rbuf_with_device(i, b);
                }
            }
        }
        t2 = MPI_Wtime();
        time = (t2 - t1) / ncycles;
        time_calc = total_ctime / ncycles;
        tover_comm = total_tover_comm / ncycles;
        double calc_slowdown_ratio = total_calc_slowdown_ratio / ncycles;
        double time_comm = time - time_calc;
        tover_calc = time_comm * calc_slowdown_ratio;
        MPI_Barrier(MPI_COMM_WORLD);
        free(requests);
        results[count] = result { true, time, time_comm + tover_comm, tover_calc, ncycles };
        return true;
    }

    DECLARE_INHERITED(AsyncBenchmark_pt2pt, sync_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_ipt2pt, async_pt2pt)
}
