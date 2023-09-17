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
#include "async_benchmark.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_topology.h"
#include "async_yaml.h"
#include "async_cuda.h"

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


    void AsyncBenchmark::init() {
        GET_PARAMETER(std::vector<int>, len);
        GET_PARAMETER(MPI_Datatype, datatype);
        GET_PARAMETER(gpu_mode_t, gpu_mode);
        is_gpu = (gpu_mode == gpu_mode_t::OFF ? false : true);
        is_cuda_aware = (gpu_mode == gpu_mode_t::CUDAAWARE);
        scope = std::make_shared<VarLenScope>(len);
        MPI_Type_size(datatype, &dtsize);
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    size_t AsyncBenchmark::get_recv_bufsize_for_len(size_t len) {
         return len * (size_t)dtsize * buf_size_multiplier_recv();
    }

    size_t AsyncBenchmark::get_send_bufsize_for_len(size_t len) {
         return len * (size_t)dtsize * buf_size_multiplier_send();
    }

    void AsyncBenchmark::alloc() {
        GET_PARAMETER(sys::host_alloc_t, host_alloc_mode);
        size_t size_to_alloc_send = get_send_bufsize_for_len(scope->get_max_len());
        size_t size_to_alloc_recv = get_recv_bufsize_for_len(scope->get_max_len());
        if (size_to_alloc_send <= ASSUMED_CACHE_SIZE * 3)
            size_to_alloc_send = ASSUMED_CACHE_SIZE * 3;
        if (size_to_alloc_recv <= ASSUMED_CACHE_SIZE * 3)
            size_to_alloc_recv = ASSUMED_CACHE_SIZE * 3;
        sys::host_mem_alloc(host_sbuf, size_to_alloc_send, host_alloc_mode);
        sys::host_mem_alloc(host_rbuf, size_to_alloc_recv, host_alloc_mode);
        if (host_rbuf == nullptr || host_sbuf == nullptr)
            throw std::runtime_error("AsyncBenchmark: memory allocation error.");
        if (is_gpu) {
            sys::device_mem_alloc(device_sbuf, size_to_alloc_send, sys::device_alloc_t::DA_CUDA);
            sys::device_mem_alloc(device_rbuf, size_to_alloc_recv, sys::device_alloc_t::DA_CUDA);
            if (device_rbuf == nullptr || device_sbuf == nullptr)
                throw std::runtime_error("AsyncBenchmark: device memory allocation error.");
        }
        allocated_size_send = size_to_alloc_send;
        allocated_size_recv = size_to_alloc_recv;
    }

    AsyncBenchmark::~AsyncBenchmark() {
    }

    void AsyncBenchmark_pt2pt::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("pt2pt"), np, rank);
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_ipt2pt::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        calc.init();
        topo = topohelper::create(p.get("pt2pt"), np, rank);
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_na2a::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("na2a"), np, rank);
        auto sources = topo->ranks_to_send_to();
        auto dests = topo->ranks_to_recv_from();
        comm_size = std::max(sources.size(), dests.size());
        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                       sources.size(), sources.data(), (const int *)MPI_UNWEIGHTED,
                                       dests.size(), dests.data(), (const int *)MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, true,
                                       &graph_comm);
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_ina2a::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        calc.init();
        topo = topohelper::create(p.get("na2a"), np, rank);
        auto sources = topo->ranks_to_send_to();
        auto dests = topo->ranks_to_recv_from();
        comm_size = std::max(sources.size(), dests.size());
        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                       sources.size(), sources.data(), (const int *)MPI_UNWEIGHTED,
                                       dests.size(), dests.data(), (const int *)MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, true,
                                       &graph_comm);
        AsyncBenchmark::alloc();
    }
   
    void AsyncBenchmark_rma_pt2pt::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("rma_pt2pt"), np, rank);
        auto sources = topo->ranks_to_send_to();
        auto dests = topo->ranks_to_recv_from();
        comm_size = std::max(sources.size(), dests.size());
        AsyncBenchmark::alloc();
        MPI_Win_create(get_sbuf(0, 0), allocated_size_send, dtsize, MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win_send);
        MPI_Win_create(get_rbuf(0, 0), allocated_size_recv, dtsize, MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win_recv);
    }

    void AsyncBenchmark_rma_ipt2pt::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        calc.init();
        topo = topohelper::create(p.get("rma_pt2pt"), np, rank);
        auto sources = topo->ranks_to_send_to();
        auto dests = topo->ranks_to_recv_from();
        comm_size = std::max(sources.size(), dests.size());
        AsyncBenchmark::alloc();
        MPI_Win_create(get_sbuf(0, 0), allocated_size_send, dtsize, MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win_send);
        MPI_Win_create(get_rbuf(0, 0), allocated_size_recv, dtsize, MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win_recv);
    }

    void AsyncBenchmark_allreduce::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("allreduce"), np, rank);
        int colour = (topo->is_active() ? topo->get_group() : MPI_UNDEFINED);
        int key = rank;
        if (coll_comm != MPI_COMM_NULL)
            MPI_Comm_split(MPI_COMM_WORLD, colour, key, &coll_comm); 
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_iallreduce::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        calc.init();
        topo = topohelper::create(p.get("allreduce"), np, rank);
        int colour = (topo->is_active() ? topo->get_group() : MPI_UNDEFINED);
        int key = rank;
        if (coll_comm != MPI_COMM_NULL)
            MPI_Comm_split(MPI_COMM_WORLD, colour, key, &coll_comm); 
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_alltoall::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        topo = topohelper::create(p.get("alltoall"), np, rank);
        int colour = (topo->is_active() ? topo->get_group() : MPI_UNDEFINED);
        int key = rank;
        MPI_Comm_split(MPI_COMM_WORLD, colour, key, &coll_comm); 
        comm_size = 1;
        if (coll_comm != MPI_COMM_NULL)
            MPI_Comm_size(coll_comm, &comm_size);
        AsyncBenchmark::alloc();
    }

    void AsyncBenchmark_ialltoall::init() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        AsyncBenchmark::init();
        calc.init();
        topo = topohelper::create(p.get("alltoall"), np, rank);
        int colour = (topo->is_active() ? topo->get_group() : MPI_UNDEFINED);
        int key = rank;
        MPI_Comm_split(MPI_COMM_WORLD, colour, key, &coll_comm); 
        comm_size = 1;
        if (coll_comm != MPI_COMM_NULL)
            MPI_Comm_size(coll_comm, &comm_size);
        AsyncBenchmark::alloc();
    }

    char *AsyncBenchmark::get_sbuf(size_t i, size_t exch_size, size_t block) { 
        size_t n = allocated_size_send / buf_size_multiplier_send() / exch_size;
        size_t block_off = allocated_size_send / buf_size_multiplier_send() * block;
        size_t off = (i%n) * exch_size; 
        if (!is_cuda_aware) {
            return host_sbuf + block_off + off;
        } else {
            return device_sbuf + block_off + off;
        }
    }

    char *AsyncBenchmark::get_rbuf(size_t i, size_t exch_size, size_t block) { 
        size_t n = allocated_size_recv / buf_size_multiplier_recv() / exch_size;
        size_t block_off = allocated_size_recv / buf_size_multiplier_recv() * block;
        size_t off = (i%n) * exch_size; 
        if (!is_cuda_aware) {
            return host_rbuf + block_off + off;
        } else {
            return device_rbuf + block_off + off;
        }
    }

    void AsyncBenchmark::sync_sbuf_with_device(size_t i, size_t exch_size) {
        if (!is_gpu)
            return;
        if (!is_cuda_aware) {
#ifdef WITH_CUDA
            auto off = get_sbuf(size_t i, size_t exch_size) - (char *)host_sbuf;
            sys::cuda::d2h_transfer((char *)host_sbuf + off, (char *)device_sbuf + off, exch_size * bufsize_multiplier_send());
#else
            (void)i;
            (void)exch_size;
#endif
        }
    }

    void AsyncBenchmark::sync_rbuf_with_device(size_t i, size_t exch_size) { 
        if (!is_gpu)
            return;
        if (!is_cuda_aware) {
#ifdef WITH_CUDA            
            auto off = get_rbuf(size_t i, size_t exch_size) - (char *)host_rbuf;
            sys::cuda::h2d_transfer((char *)device_rbuf + off, host_rbuf + off, exch_size * bufsize_multiplier_recv());
#else
            (void)i;
            (void)exch_size;
#endif            
        }
    }

    void AsyncBenchmark::run(const scope_item &item) { 
        GET_PARAMETER(MPI_Datatype, datatype);
        GET_PARAMETER(std::vector<int>, ncycles);
        GET_PARAMETER(std::vector<int>, len);
        GET_PARAMETER(int, nwarmup);
        if (len.size() == 0) {
            throw std::runtime_error("AsyncBenchmark: wrong len parameter");
        }
        if (ncycles.size() == 0) {
            throw std::runtime_error("AsyncBenchmark: wrong ncycles parameter");
        }
        int item_ncycles = ncycles[0];
        for (size_t i =0; i < len.size(); i++) {
            if (item.len == (size_t)len[i]) {
                item_ncycles = (i >= ncycles.size() ? ncycles.back() : ncycles[i]);
            }
        }
        double time, tover_comm, tover_calc;
        bool done = benchmark(item.len, datatype, nwarmup, item_ncycles, time, tover_comm, tover_calc);
        if (!done) {
            results[item.len] = result { false, 0.0, 0.0, 0.0, item_ncycles };
        }
    }

    void AsyncBenchmark::finalize() { 
        GET_PARAMETER(YAML::Emitter, yaml_out);
        GET_PARAMETER(std::string, yaml_outfile);
        YamlOutputMaker yaml_tmin("tmin");
        YamlOutputMaker yaml_tmax("tmax");
        YamlOutputMaker yaml_tavg("tavg");
        YamlOutputMaker yaml_over_comm("over_comm");
        YamlOutputMaker yaml_over_calc("over_calc");
        YamlOutputMaker yaml_over_full("over_full");
        YamlOutputMaker yaml_topo("topo");
        for (auto it = results.begin(); it != results.end(); ++it) {
            int len = it->first;
            double time = (it->second).time, tmin = 0, tmax = 0, tavg = 0;
            double tover_comm = 0, tover_calc = 0;
            int is_done = ((it->second).done ? 1 : 0), nexec = 0;
            MPI_Reduce(&is_done, &nexec, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!(it->second).done) time = 1e32;
            MPI_Reduce(&time, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            if (!(it->second).done) time = 0.0;
            MPI_Reduce(&time, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            tavg = get_avg(time, nexec, rank, np, is_done); 
            tover_comm = get_avg((it->second).overhead_comm, nexec, rank, np, is_done);
            tover_calc = get_avg((it->second).overhead_calc, nexec, rank, np, is_done);
            if (rank == 0) {
                if (nexec == 0) {
                    std::cout << get_name() << ": " << "{ " << "len: " << len << ", "
                        << " error: \"no successful executions!\"" << " }" << std::endl;
                } else {
                    std::cout << get_name() << ": " << "{ " << "len: " << len << ", "
                        << "ncycles: " << (it->second).ncycles << ", "
                        << " time: [ " << tavg << ", " 
                                      << tmin << ", " 
                                      << tmax << " ]" 
                        << ", overhead: [ " << tover_comm << " , " << tover_calc 
                                      << " ] }" << std::endl;
                    yaml_tmin.add(len, tmin);
                    yaml_tmax.add(len, tmax);
                    yaml_tavg.add(len, tavg);
                    yaml_over_comm.add(len, tover_comm); 
                    yaml_over_calc.add(len, tover_calc); 
                    yaml_over_full.add(len, tover_calc + tover_comm);
                }
            }
        }
        if (!yaml_outfile.empty()) {
            yaml_topo.add("name", topo->name());
            yaml_topo.add("np", np);
            for (int n = 0; n < np; n++) {
                auto per_rank_topo = topo->clone(n);
                std::vector<double> ranks;
                for (auto &action : per_rank_topo->comm_actions()) {
                    ranks.push_back(action.rank);
                }
                yaml_topo.add(n, ranks);
            } 
            WriteOutYaml(yaml_out, get_name(), {yaml_tavg, yaml_tmin, yaml_tmax, yaml_over_full, yaml_over_comm, yaml_over_calc, yaml_topo});
        }

        // NOTE: can't free pinned memory in destructor, CUDA runtime complains
        // that it's too late
        sys::host_mem_free(host_sbuf, host_alloc_mode);
        sys::host_mem_free(host_rbuf, host_alloc_mode);
        sys::device_mem_free(device_sbuf, sys::device_alloc_t::DA_CUDA);
        sys::device_mem_free(device_rbuf, sys::device_alloc_t::DA_CUDA);
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

    bool AsyncBenchmark_allreduce::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                             double &time, double &tover_comm, double &tover_calc) {
        time = 0;
        tover_comm = 0;
        tover_calc = 0;

        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0;
        if (topo->is_active()) {
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i >= nwarmup) t1 = MPI_Wtime();
                sync_sbuf_with_device(i, b);
                MPI_Allreduce(get_sbuf(i, b), get_rbuf(i, b), count, datatype, MPI_SUM, coll_comm);
                sync_rbuf_with_device(i, b);
                if (i >= nwarmup) {
                    t2 = MPI_Wtime();
                    time += (t2 - t1);
                }
                barrier(rank, np, coll_comm);
#if ASYNC_EXTRA_BARRIER
                barrier(rank, np, coll_comm);
                barrier(rank, np, coll_comm);
                barrier(rank, np, coll_comm);
                barrier(rank, np, coll_comm);
#endif
            }
        }
        time /= ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0, 0, ncycles };
        return true;
    }

    bool AsyncBenchmark_iallreduce::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                              double &time, double &tover_comm, double &tover_calc) {

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
            MPI_Iallreduce(get_sbuf(i, b), get_rbuf(i, b), count, datatype, MPI_SUM, coll_comm, request);
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
            barrier(rank, np);
#if ASYNC_EXTRA_BARRIER
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
#endif
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

    bool AsyncBenchmark_alltoall::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                             double &time, double &tover_comm, double &tover_calc) {
        time = 0;
        tover_comm = 0;
        tover_calc = 0;
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0;
        if (topo->is_active()) {
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i >= nwarmup) t1 = MPI_Wtime();
                sync_sbuf_with_device(i, b);
                MPI_Alltoall(get_sbuf(i, b), count, datatype, get_rbuf(i, b), count, datatype, coll_comm);
                sync_rbuf_with_device(i, b);
                if (i >= nwarmup) {
                    t2 = MPI_Wtime();
                    time += (t2 - t1);
                }
                barrier(rank, np, coll_comm);
#if ASYNC_EXTRA_BARRIER
                barrier(rank, np, coll_comm);
                barrier(rank, np, coll_comm);
                barrier(rank, np, coll_comm);
                barrier(rank, np, coll_comm);
#endif
            }
        }
        time /= ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0, 0, ncycles };
        return true;
    }

    bool AsyncBenchmark_ialltoall::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                              double &time, double &tover_comm, double &tover_calc) {
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
            MPI_Ialltoall(get_sbuf(i, b), count, datatype, get_rbuf(i, b), count, datatype, coll_comm, request);
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
            barrier(rank, np);
#if ASYNC_EXTRA_BARRIER
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
#endif
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

    bool AsyncBenchmark_na2a::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                        double &time, double &tover_comm, double &tover_calc) {          
        time = 0;
        tover_comm = 0;
        tover_calc = 0;
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
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
            barrier(rank, np);
#if ASYNC_EXTRA_BARRIER
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
#endif
        }
        time /= ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time, 0, 0, ncycles };
        return true;
    }

    bool AsyncBenchmark_ina2a::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                         double &time, double &tover_comm, double &tover_calc) {         
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0, time_calc = 0, total_ctime = 0, total_tover_comm = 0, total_calc_slowdown_ratio = 0,
                                              local_ctime = 0, local_tover_comm = 0, local_calc_slowdown_ratio = 0;
        time = 0;
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
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
            barrier(rank, np);
#if ASYNC_EXTRA_BARRIER
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
#endif
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

    bool AsyncBenchmark_rma_pt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                             double &time, double &tover_comm, double &tover_calc) {
        tover_comm = 0;
        tover_calc = 0;
        if (!topo->is_active()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = get_send_bufsize_for_len(count);
        double t1 = 0, t2 = 0;
        auto comm_actions = topo->comm_actions();
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i == nwarmup) t1 = MPI_Wtime();
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int ns = 0, nr = 0;
                int rank = comm_actions[commstage].rank;
                size_t off = (get_sbuf(i, b) - get_sbuf(0, 0)) / dtsize;
                sync_sbuf_with_device(i, b);
                if (comm_actions[commstage].action == action_t::RECV) {
                    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win_recv);
                    MPI_Get(get_rbuf(i, b, nr++), count, datatype, rank, off, count, datatype, win_recv);
                    MPI_Win_unlock(rank, win_recv);
                } else if (comm_actions[commstage].action == action_t::SEND) {
                    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win_send);
                    MPI_Put(get_sbuf(i, b, ns++), count, datatype, rank, off, count, datatype, win_send);
                    MPI_Win_unlock(rank, win_send);
                }
                sync_rbuf_with_device(i, b);
            }
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
            void init(const std::vector<int> &ranks) {
                for (auto r : ranks) {
                    lock_ranks[r] = status_and_num_t {false, 0};
                }
            }
            int lock_if_needed(int r, MPI_Win &win, bool *result_ptr = nullptr) {
                auto &status_and_num = lock_ranks[r];
                if (!status_and_num.first) {
                    MPI_Win_lock(MPI_LOCK_SHARED, r, 0, win);
                    lock_ranks[r] = status_and_num_t { true, n };
                    if (result_ptr)
                        *result_ptr = true;
                    return n++;
                }
                if (result_ptr)
                    *result_ptr = false;
                return status_and_num.second;
            }
            int unlock_all(MPI_Win &win) {
                for (auto &it : lock_ranks) {
                    auto &status_and_num = it.second;
                    if (status_and_num.first) {
                        MPI_Win_unlock(it.first, win);
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
        for (int i = 0; i < ncycles + nwarmup; i++) {
            size_t off = (get_sbuf(i, b) - get_sbuf(0, 0)) / dtsize;
            sendinfo.init(ranks_recv);
            recvinfo.init(ranks_send);
            if (i == nwarmup) t1 = MPI_Wtime();
            int nr = 0;
            sync_sbuf_with_device(i, b);
            for (size_t commstage = 0; commstage < comm_actions.size(); commstage++) {
                int r = comm_actions[commstage].rank;
                if (comm_actions[commstage].action == action_t::RECV) {
                    int n = sendinfo.lock_if_needed(r, win_recv);
                    MPI_Rget(get_rbuf(i, b, n), count, datatype, r, off, 
                             count, datatype, win_recv, &requests[nr++]);
                } else if (comm_actions[commstage].action == action_t::SEND) {
                    int n = sendinfo.lock_if_needed(r, win_send);
                    MPI_Rput(get_sbuf(i, b, n), count, datatype, r, off, 
                             count, datatype, win_send, &requests[nr++]);
                }
            }
            calc.benchmark(count, datatype, 0, 1, local_ctime, local_tover_comm, local_calc_slowdown_ratio);
            MPI_Waitall(calc.num_requests, requests, MPI_STATUSES_IGNORE);
            sendinfo.unlock_all(win_send);
            recvinfo.unlock_all(win_recv);
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

    void AsyncBenchmark_calc::gpu_calc_loop() {
#ifdef WITH_CUDA                
        if (!is_gpu_calculations)
            return;
        while (true) {
            if (gpu_calc_cycle_active && sys::cuda::is_device_idle()) {
                //std::cout << ">> rank=" << rank << " submit GPU workload; gpu_workload_calibration=" << gpu_workload_calibration << std::endl;
                for (int i = 0; i < 5; i++) {
                    sys::cuda::submit_workload(gpu_workload_ncycles, gpu_workload_calibration);
                    if (gpu_workload_transfer_size != 0) {
                        sys::cuda::d2h_transfer(host_transfer_buf, device_transfer_buf, gpu_workload_transfer_size,
                                                sys::cuda::transfer_t::WORKLOAD);
                    }
                }
            } else {
                if (gpu_calc_cycle_finish)
                    return;
                usleep(100);
            }
        }
#endif        
    }

    // NOTE: to ensure just calc, no manual progress call it with iters_till_test == R
    // NOTE2: tover_comm is not zero'ed here before operation!
    void AsyncBenchmark_calc::calc_and_progress_loop(int ncalccycles, int iters_till_test, double &tover_comm) {
        gpu_calc_cycle_active = true;
        if (is_cpu_calculations) { 
            for (volatile int repeat = 0, cnt = iters_till_test; repeat < ncalccycles; repeat++) {
                if (cnt-- == 0) { 
                    double t1 = MPI_Wtime();
                    if (reqs && num_requests) {
                        for (int r = 0; r < num_requests; r++) {
                            if (!stat[r]) {
                                total_tests++;
                                MPI_Test(&reqs[r], &stat[r], MPI_STATUS_IGNORE);
                                if (stat[r]) {
                                    successful_tests++;
                                }
                            }
                        }
                    }
                    double t2 = MPI_Wtime();
                    tover_comm += (t2 - t1);
                    cnt = iters_till_test;
                } 
                for (volatile size_t i = 0; i < CALC_MATRIX_SIZE; i++) {
                    for (volatile size_t j = 0; j < CALC_MATRIX_SIZE; j++) {
                        for (volatile size_t k = 0; k < CALC_MATRIX_SIZE; k++) {
                            c[i][j] += a[i][k] * b[k][j] + repeat*repeat;
                        }
                    }
                }
            }
        } else {
            usleep(ncalccycles * 10 / cycles_per_10usec);
        }
        gpu_calc_cycle_active = false;
    }

    void AsyncBenchmark_calc::calc_loop(int ncalccycles, double &tover_comm) {
        calc_and_progress_loop(ncalccycles, ncalccycles, tover_comm);
    }

    void AsyncBenchmark_calc::calc_loop(int ncalccycles) {
        double tover_comm;
        calc_and_progress_loop(ncalccycles, ncalccycles, tover_comm);
    }

    void AsyncBenchmark_calc::calibration() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        is_cpu_calculations = true;
        is_gpu_calculations = false;
        is_omit_calculation_loop = false;
        int estcycles = p.get("calc_calibration").get_int("estimation_cycles");
        double timings[3];
        if (estcycles == 0) {
            throw std::runtime_error("AsyncBenchmark_calc: either -cycles_per_10usec or -estcycles option is required.");
        }
        int Nrep = (int)(2000000ul / (unsigned long)(CALC_MATRIX_SIZE*CALC_MATRIX_SIZE*CALC_MATRIX_SIZE));
        for (int k = 0; k < 3 + estcycles; k++) {
            double t1 = MPI_Wtime();
            calc_loop(Nrep);
            double t2 = MPI_Wtime();
            if (k >= estcycles)
                timings[k - estcycles] = t2 - t1;
            else {
                if (k > 0) {
                    if (t2 - t1 > 1.5) {
                        Nrep = (int)((double)Nrep * 1.0 / (t2 - t1));
                    } else if (t2 - t1 > 0.001 && t2 - t1 < 0.5) {
                        Nrep = (int)((double)Nrep * 1.0 / (t2 - t1));
                    } else if (t2 - t1 < 0.001) {
                        std::cout << ">> ERROR: cycles_per_10usec: too little measuring time."
                            << " You may increase CALC_MATRIX_SIZE constant." << std::endl;
                        throw std::runtime_error("cycles_per_10usec: calibration cycle error: too little measuring time");
                    }
                }
            }
        }
        double tmedian = std::min(timings[0], timings[1]);
        if (tmedian < timings[2])
            tmedian = std::min(std::max(timings[0], timings[1]), timings[2]);
        double _10usec = 1.0e5;
        int local_cycles_per_10usec = (int)((double)Nrep / (tmedian * _10usec) + 0.999);
        MPI_Allreduce(&local_cycles_per_10usec, &cycles_per_10usec_avg, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_cycles_per_10usec, &cycles_per_10usec_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_cycles_per_10usec, &cycles_per_10usec_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        cycles_per_10usec_avg /= np;
        if (cycles_per_10usec_avg < 150 && cycles_per_10usec_avg > 10) {
            int hits = 0;
            int local_hit = ((fabs((float)local_cycles_per_10usec - (float)cycles_per_10usec_avg) > 
                        (float)cycles_per_10usec_avg/25.0) ? 0 : 1);
            MPI_Allreduce(&local_hit, &hits, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if ((float)(np - hits) / (float)np > 0.1f) {
                if (rank == 0) {
                    std::cout << ">> cycles_per_10usec: WARNING: many deviated values!" << std::endl;
                    irregularity_level++;
                }
            }
            if (cycles_per_10usec_min == 0 || hits == 0) {
                if (rank == 0) {
                    std::cout << ">> cycles_per_10usec: WARNING: very strange and deviated calibration results" << std::endl;
                    irregularity_level += 2;
                }
            } else if (cycles_per_10usec_max / cycles_per_10usec_min >= 4 && cycles_per_10usec_avg / cycles_per_10usec_min >= 2) {
                // exclude highly deviated values
                int cleaned_local_cycles_per_10usec = (local_hit ? local_cycles_per_10usec : 0);
                cycles_per_10usec_avg = 0;
                MPI_Allreduce(&cleaned_local_cycles_per_10usec, &cycles_per_10usec_avg, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                cycles_per_10usec_avg /= hits;
                irregularity_level++;
            }
        }
#if 0                
        char node[80];
        gethostname(node, 80-1);
        std::cout << ">> cycles_per_10usec: node: " << node << " time=" << tmedian << "; cpersec=" << (double)Nrep/tmedian << std::endl;
        std::cout << ">> cycles_per_10usec: node: " << node << " cycles_per_10usec=" << cycles_per_10usec_avg << std::endl;
#endif                
        if (rank == 0) {
            std::cout << ">> " << get_name() << ": average cycles_per_10usec=" << cycles_per_10usec_avg << " min/max=" 
                << cycles_per_10usec_min << "/" << cycles_per_10usec_max << std::endl;
            if (cycles_per_10usec_avg > 150 || cycles_per_10usec_avg < 10) {
                irregularity_level++;
                std::cout << ">> cycles_per_10usec: NOTE: good value for cycles_per_10usec is [10, 150]."
                    << " You may decrease or increase CALC_MATRIX_SIZE constant." << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void AsyncBenchmark_calc::init() {
        AsyncBenchmark::init();
        GET_PARAMETER(std::vector<int>, calctime);
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        if (p.find("workload")) {
            is_cpu_calculations = p.get("workload").get_bool("cpu_calculations");
            is_gpu_calculations = p.get("workload").get_bool("gpu_calculations");
            if (!is_cpu_calculations && !is_gpu_calculations) {
                is_omit_calculation_loop = true;
            }
            if (is_gpu_calculations && !is_gpu) {
                throw std::runtime_error("GPU workload for calculations requires GPU mode.");
            }
            if (is_cpu_calculations) {
                is_manual_progress = p.get("workload").get_bool("manual_progress");
                cycles_per_10usec = p.get("workload").get_int("cycles_per_10usec");
                is_omit_calulation_overhead_estimation = p.get("workload").get_bool("omit_calc_over_est");
            } else {
                if (p.get("workload").get_bool("manual_progress")) {
                    throw std::runtime_error("Manual progress works only when CPU calculations are switched on.");
                }
                is_manual_progress = false;
            }
            for (size_t i = 0; i < len.size(); i++) {
                calctime_by_len[len[i]] = (i >= calctime.size() ? (calctime.size() == 0 ? 10000 : calctime[calctime.size() - 1]) : calctime[i]);
            }            
        } else {
            is_omit_calculation_loop = true;
            is_cpu_calculations = false;
            is_gpu_calculations = false;
            is_manual_progress = false;
        }
        if (is_gpu_calculations) {
            is_gpu_calculations = sys::is_it_the_rank_for_gpu_calc();
        }
        for (size_t i = 0; i < CALC_MATRIX_SIZE; i++) {
            x[i] = y[i] = 0.;
            for (size_t j=0; j < CALC_MATRIX_SIZE; j++) {
                a[i][j] = 1.;
            }
        }
#ifdef WITH_CUDA       
        if (is_gpu_calculations) {
            gpu_workload_calibration = sys::cuda::workload_calibration(); 
            sys::host_mem_alloc(host_transfer_buf, gpu_workload_transfer_size, host_alloc_mode);
            if (host_transfer_buf == nullptr) {
                throw std::runtime_error("AsyncBenchmark: memory allocation error.");
            }
            sys::device_mem_alloc(device_transfer_buf, gpu_workload_transfer_size, sys::device_alloc_t::DA_CUDA);
            if (device_transfer_buf == nullptr) {
                throw std::runtime_error("AsyncBenchmark: memory allocation error.");
            }
            gpu_calc_cycle_active = false;
            gpu_calc_cycle_finish = false;
            p_tgpucalc = new std::thread([&]() {
                gpu_calc_loop();
            });
        }
#endif        
    }

    void AsyncBenchmark_calc::finalize() {
#ifdef WITH_CUDA
        if (p_tgpucalc != nullptr) {
            gpu_calc_cycle_active = false;
            gpu_calc_cycle_finish = true;
            p_tgpucalc->join();
            sys::cuda::sync_contexts();
            gpu_calc_cycle_finish = false;
        }
#endif
        AsyncBenchmark::finalize();
    }

    bool AsyncBenchmark_calc::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                        double &time, double &tover_comm, double &tover_calc) {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        int real_cycles_per_10usec;
        (void)datatype;
        total_tests = 0;
        successful_tests = 0;
        time = 0;
        tover_comm = 0;
        tover_calc = 0;
        if (is_omit_calculation_loop) {
            time = 0;
            return true;
        }
        double t1 = 0, t2 = 0;
        
        if (cycles_per_10usec == 0) {
            if (is_cpu_calculations && !is_omit_calulation_overhead_estimation) {
                throw std::runtime_error("Wrong cycles_per_10usec constant value.");
            }
            cycles_per_10usec = 10;
        }
        int ncalccycles = calctime_by_len[count] * cycles_per_10usec / 10;
        if (is_manual_progress && reqs) {
            for (int r = 0; r < num_requests; r++) {
                stat[r] = 0;
            }
        }
        if (is_manual_progress) {
       	    int spinperiod = p.get("workload").get_int("spin_period");
            const int cnt_for_mpi_test = std::max(spinperiod * cycles_per_10usec / 10, 1);
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) 
                    t1 = MPI_Wtime();
                calc_and_progress_loop(ncalccycles, cnt_for_mpi_test, tover_comm);
            }
        } else {
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) 
                    t1 = MPI_Wtime();
                calc_loop(ncalccycles, tover_comm);
            }
        }
        t2 = MPI_Wtime();
        time = (t2 - t1);
        if (is_cpu_calculations && !is_omit_calulation_overhead_estimation) {
            int pure_calc_time_in_usec = int((time - tover_comm) * 1e6);
            if (!pure_calc_time_in_usec)
                return true;
            real_cycles_per_10usec = ncalccycles * 10 / pure_calc_time_in_usec;
            if (cycles_per_10usec && real_cycles_per_10usec) {
                int ncalccycles_expected = pure_calc_time_in_usec * cycles_per_10usec / 10;
                double ratio = 0;
                if (ncalccycles < ncalccycles_expected) {
                    ratio = (double)ncalccycles_expected / (double)ncalccycles - 1.0;
                }
                tover_calc = ratio;
            }
        } else {
            tover_calc = 0;
        }
        return true;
    }

    void AsyncBenchmark_calibration::init() {
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        scope = std::make_shared<VarLenScope>(0, 0);
        calc.init();
    }

    void AsyncBenchmark_calibration::run(const scope_item &item) {
        (void)item;
        calc.calibration();
    }

    static inline bool isregular(int l, int g, int np) {
        int regular = 0, local_regular = 0;
        local_regular = (l == g ? 1 : 0);
        MPI_Allreduce(&local_regular, &regular, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return regular == np;
    }

    void AsyncBenchmark_calibration::finalize() {
        GET_PARAMETER(YAML::Emitter, yaml_out);
        GET_PARAMETER(std::string, yaml_outfile);
        
        int isset = 0, ncores = 0, nthreads = 0;
        int local_isset = 0, local_ncores = 0, local_nthreads = 0;
        local_isset = (int)sys::threadaffinityisset(local_nthreads);
        local_ncores = sys::getnumcores();
        MPI_Allreduce(&local_isset, &isset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        isset /= np;
        MPI_Allreduce(&local_ncores, &ncores, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        ncores /= np;
        MPI_Allreduce(&local_nthreads, &nthreads, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        nthreads /= np;
        isset = (isregular(local_isset, isset, np) ? isset : -1);
        ncores = (isregular(local_ncores, ncores, np) ? ncores : -1);
        nthreads = (isregular(local_nthreads, nthreads, np) ? nthreads : -1);
        if (!yaml_outfile.empty()) {
            YamlOutputMaker yaml_affinity("affinity");
            yaml_affinity.add("isset", isset);
            yaml_affinity.add("ncores", ncores);
            yaml_affinity.add("nthreads", nthreads);
            
            YamlOutputMaker yaml_calibration("calibration");
            yaml_calibration.add("avg", calc.cycles_per_10usec_avg); 
            yaml_calibration.add("min", calc.cycles_per_10usec_min); 
            yaml_calibration.add("max", calc.cycles_per_10usec_max); 
            yaml_calibration.add("irregularity", calc.irregularity_level); 

            WriteOutYaml(yaml_out, get_name(), {yaml_affinity, yaml_calibration});
        }
    }


    DECLARE_INHERITED(AsyncBenchmark_pt2pt, sync_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_ipt2pt, async_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_allreduce, sync_allreduce)
    DECLARE_INHERITED(AsyncBenchmark_iallreduce, async_allreduce)
    DECLARE_INHERITED(AsyncBenchmark_alltoall, sync_alltoall)
    DECLARE_INHERITED(AsyncBenchmark_ialltoall, async_alltoall)
    DECLARE_INHERITED(AsyncBenchmark_na2a, sync_na2a)
    DECLARE_INHERITED(AsyncBenchmark_ina2a, async_na2a)
    DECLARE_INHERITED(AsyncBenchmark_rma_pt2pt, sync_rma_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_rma_ipt2pt, async_rma_pt2pt)
    DECLARE_INHERITED(AsyncBenchmark_calc, workload)
    DECLARE_INHERITED(AsyncBenchmark_calibration, calc_calibration)
}
