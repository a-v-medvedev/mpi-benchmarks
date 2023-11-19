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
#include <vector>
#include "async_benchmark.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_params.h"
#include "async_topology.h"
#include "async_yaml.h"
#include "async_cuda.h"

namespace async_suite {
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
        if (!topo->is_active()) {
            return;
        }
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
   
    char *AsyncBenchmark::get_sbuf() { 
        if (!is_cuda_aware) {
            return host_sbuf;
        } else {
            return device_sbuf;
        }
    }

    char *AsyncBenchmark::get_rbuf() { 
        if (!is_cuda_aware) {
            return host_rbuf;
        } else {
            return device_rbuf;
        }
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
        GET_PARAMETER(sys::host_alloc_t, host_alloc_mode);
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
}
