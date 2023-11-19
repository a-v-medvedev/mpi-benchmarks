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
#include "async_na2a.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_topology.h"
#include "async_yaml.h"
#include "async_cuda.h"

namespace async_suite {
    void AsyncBenchmark_na2a_base::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("na2a"), np, rank);
        auto sources = topo->ranks_to_send_to();
        auto dests = topo->ranks_to_recv_from();
        comm_size = std::max(sources.size(), dests.size());
        AsyncBenchmark::alloc();
        if (topo->is_active()) {
            MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                           sources.size(), sources.data(), (const int *)MPI_UNWEIGHTED,
                                           dests.size(), dests.data(), (const int *)MPI_UNWEIGHTED,
                                           MPI_INFO_NULL, true,
                                           &graph_comm);
        } else {
            MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                           0, nullptr, (const int *)MPI_UNWEIGHTED,
                                           0, nullptr, (const int *)MPI_UNWEIGHTED,
                                           MPI_INFO_NULL, true,
                                           &graph_comm);
           
        }
    }

    void AsyncBenchmark_na2a::init() {
        AsyncBenchmark_na2a_base::init();
    }

    void AsyncBenchmark_ina2a::init() {
        AsyncBenchmark_na2a_base::init();
        calc.init();
    }
   
    bool AsyncBenchmark_na2a::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                        double &time, double &tover_comm, double &tover_calc) {          
        time = 0;
        tover_comm = 0;
        tover_calc = 0;
        if (!topo->is_active()) {
            for (int i = 0; i < ncycles + nwarmup; i++) {
                barrier(rank, np, graph_comm);
                if (EXTRA_BARRIER) {
                    barrier(rank, np, graph_comm);
                    barrier(rank, np, graph_comm);
                    barrier(rank, np, graph_comm);
                    barrier(rank, np, graph_comm);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        assert(graph_comm != MPI_COMM_NULL);
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0;
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i >= nwarmup) t1 = MPI_Wtime();
            sync_sbuf_with_device(i, b);
            MPI_Neighbor_alltoall(get_sbuf(i, b), count, datatype,
                                  get_rbuf(i, b), count, datatype,
                                  graph_comm);            
            sync_rbuf_with_device(i, b);
            if (i >= nwarmup) {
                t2 = MPI_Wtime();
                time += (t2 - t1);
            }
            barrier(rank, np, graph_comm);
            if (EXTRA_BARRIER) {
                barrier(rank, np, graph_comm);
                barrier(rank, np, graph_comm);
                barrier(rank, np, graph_comm);
                barrier(rank, np, graph_comm);
            }
        }
        time /= ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0, 0, ncycles };
        return true;
    }

    bool AsyncBenchmark_ina2a::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                         double &time, double &tover_comm, double &tover_calc) {         
        if (!topo->is_active()) {
            for (int i = 0; i < ncycles + nwarmup; i++) {
                barrier(rank, np, graph_comm);
                if (EXTRA_BARRIER) {
                    barrier(rank, np, graph_comm);
                    barrier(rank, np, graph_comm);
                    barrier(rank, np, graph_comm);
                    barrier(rank, np, graph_comm);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        assert(graph_comm != MPI_COMM_NULL);
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0, time_calc = 0, total_ctime = 0, total_tover_comm = 0, total_calc_slowdown_ratio = 0,
                                              local_ctime = 0, local_tover_comm = 0, local_calc_slowdown_ratio = 0;
        time = 0;
        MPI_Request request[1];
        calc.reqs = request;
        calc.num_requests = 1;
        MPI_Status  status;
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i >= nwarmup) t1 = MPI_Wtime();
            sync_sbuf_with_device(i, b);
            MPI_Ineighbor_alltoall(get_sbuf(i, b), count, datatype,
                                   get_rbuf(i, b), count, datatype,
                                   graph_comm, request);
            calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_calc_slowdown_ratio);
            MPI_Wait(request, &status);
            sync_rbuf_with_device(i, b);
            if (i >= nwarmup) {
                t2 = MPI_Wtime();
                time += (t2 - t1);
                total_ctime += local_ctime;
                total_tover_comm += local_tover_comm;
                total_calc_slowdown_ratio += local_calc_slowdown_ratio;
            }
            barrier(rank, np, graph_comm);
            if (EXTRA_BARRIER) {
                barrier(rank, np, graph_comm);
                barrier(rank, np, graph_comm);
                barrier(rank, np, graph_comm);
                barrier(rank, np, graph_comm);
            }
        }
        time /= ncycles;
        time_calc = total_ctime / ncycles;
        tover_comm = total_tover_comm / ncycles;
        double calc_slowdown_ratio = total_calc_slowdown_ratio / ncycles;
        double time_comm = time - time_calc;
        tover_calc = time_comm * calc_slowdown_ratio;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, time_comm + tover_comm, tover_calc, ncycles };
        return true;
    }

    DECLARE_INHERITED(AsyncBenchmark_na2a, sync_na2a)
    DECLARE_INHERITED(AsyncBenchmark_ina2a, async_na2a)
}
