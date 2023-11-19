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
#include "async_rma.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_topology.h"
#include "async_yaml.h"
#include "async_cuda.h"

namespace async_suite {

    void AsyncBenchmark_rma_pt2pt_base::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("rma_pt2pt"), np, rank);
        auto sources = topo->ranks_to_send_to();
        auto dests = topo->ranks_to_recv_from();
        comm_size = std::max(sources.size(), dests.size());
        AsyncBenchmark::alloc();
        if (topo->is_active()) {
            MPI_Win_create(get_sbuf(), allocated_size_send, dtsize, MPI_INFO_NULL,
                           MPI_COMM_WORLD, &win_send);
            MPI_Win_create(get_rbuf(), allocated_size_recv, dtsize, MPI_INFO_NULL,
                           MPI_COMM_WORLD, &win_recv);
        }
    }

    void AsyncBenchmark_rma_pt2pt::init() {
        AsyncBenchmark_rma_pt2pt_base::init();
    }

    void AsyncBenchmark_rma_ipt2pt::init() {
        AsyncBenchmark_rma_pt2pt_base::init();
        calc.init();
    }

    bool AsyncBenchmark_rma_pt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                             double &time, double &tover_comm, double &tover_calc) {
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        tover_comm = 0;
        tover_calc = 0;
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0;
        auto comm_actions = topo->comm_actions();
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i == nwarmup) t1 = MPI_Wtime();
            sync_sbuf_with_device(i, b);
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int ns = 0, nr = 0;
                int rank = comm_actions[commstage].rank;
                size_t off = (get_sbuf(i, b) - get_sbuf()) / dtsize;
                if (comm_actions[commstage].action == action_t::RECV) {
                    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win_recv);
                    MPI_Get(get_rbuf(i, b, nr++), count, datatype, rank, off, count, datatype, win_recv);
                    MPI_Win_unlock(rank, win_recv);
                } else if (comm_actions[commstage].action == action_t::SEND) {
                    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win_send);
                    MPI_Put(get_sbuf(i, b, ns++), count, datatype, rank, off, count, datatype, win_send);
                    MPI_Win_unlock(rank, win_send);
                }
            }
            sync_rbuf_with_device(i, b);
        }
        t2 = MPI_Wtime();
        time = (t2 - t1) / ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0.0, 0.0, ncycles };
        return true;
    }

    bool AsyncBenchmark_rma_ipt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                              double &time, double &tover_comm, double &tover_calc) {
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0, time_calc = 0, total_ctime = 0, total_tover_comm = 0, total_calc_slowdown_ratio = 0,
                                              local_ctime = 0, local_tover_comm = 0, local_calc_slowdown_ratio = 0;
        auto comm_actions = topo->comm_actions();
        MPI_Request *requests;
        requests = (MPI_Request *)calloc(sizeof(MPI_Request), topo->get_num_actions());
        calc.reqs = requests;
        calc.num_requests = topo->get_num_actions();
        using status_and_num_t = std::pair<bool, int>;
        std::map<int, status_and_num_t> lock_ranks;
        struct peer_data {
            std::map<int, status_and_num_t> lock_ranks;
            size_t n = 0;
            MPI_Win *win_ptr = nullptr;
            void init(MPI_Win &win, const std::vector<int> &ranks) {
                for (auto r : ranks) {
                    lock_ranks[r] = status_and_num_t {false, 0};
                }
                win_ptr = &win;
            }
            int lock_if_needed(int r, bool *result_ptr = nullptr) {
                assert(win_ptr);
                assert(lock_ranks.find(r) != lock_ranks.end());
                auto &status_and_num = lock_ranks[r];
                if (!status_and_num.first) {
                    MPI_Win_lock(MPI_LOCK_SHARED, r, 0, *win_ptr);
                    lock_ranks[r] = status_and_num_t { true, n };
                    if (result_ptr)
                        *result_ptr = true;
                    return n++;
                }
                if (result_ptr)
                    *result_ptr = false;
                return status_and_num.second;
            }
            int unlock_all() {
                assert(win_ptr);
                for (auto &it : lock_ranks) {
                    auto &status_and_num = it.second;
                    if (status_and_num.first) {
                        MPI_Win_unlock(it.first, *win_ptr);
                        status_and_num = status_and_num_t { false, 0 };
                    }
                }
                int retval = n;
                n = 0;
                return retval;
            }
        } sendinfo, recvinfo;
        auto ranks_recv = topo->ranks_to_recv_from();
        auto ranks_send = topo->ranks_to_send_to();
        sendinfo.init(win_send, ranks_send);
        recvinfo.init(win_recv, ranks_recv);
        for (int i = 0; i < ncycles + nwarmup; i++) {
            size_t off = (get_sbuf(i, b) - get_sbuf()) / dtsize;
            if (i == nwarmup) t1 = MPI_Wtime();
            int nreq = 0;
            sync_sbuf_with_device(i, b);
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int r = comm_actions[commstage].rank;
                if (comm_actions[commstage].action == action_t::RECV) {
                    int n = recvinfo.lock_if_needed(r);
                    MPI_Rget(get_rbuf(i, b, n), count, datatype, r, off, 
                             count, datatype, win_recv, &requests[nreq++]);
                } else if (comm_actions[commstage].action == action_t::SEND) {
                    int n = sendinfo.lock_if_needed(r);
                    MPI_Rput(get_sbuf(i, b, n), count, datatype, r, off, 
                             count, datatype, win_send, &requests[nreq++]);
                }
            }
            calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_calc_slowdown_ratio);
            MPI_Waitall(calc.num_requests, requests, MPI_STATUSES_IGNORE);
            sendinfo.unlock_all();
            recvinfo.unlock_all();
            sync_rbuf_with_device(i, b);
            if (i >= nwarmup) {
                total_ctime += local_ctime;
                total_tover_comm += local_tover_comm;
                total_calc_slowdown_ratio += local_calc_slowdown_ratio;
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

    DECLARE_INHERITED(AsyncBenchmark_rma_pt2pt, sync_rma_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_rma_ipt2pt, async_rma_pt2pt)
}
