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
#include "async_workload.h"
#include "async_sys.h"
#include "async_average.h"
#include "async_topology.h"
#include "async_yaml.h"
#include "async_cuda.h"

namespace async_suite {

    void AsyncBenchmark_workload::gpu_calc_loop() {
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
    void AsyncBenchmark_workload::calc_and_progress_loop(int ncalccycles, int iters_till_test, double &tover_comm) {
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

    void AsyncBenchmark_workload::calc_loop(int ncalccycles, double &tover_comm) {
        calc_and_progress_loop(ncalccycles, ncalccycles, tover_comm);
    }

    void AsyncBenchmark_workload::calc_loop(int ncalccycles) {
        double tover_comm;
        calc_and_progress_loop(ncalccycles, ncalccycles, tover_comm);
    }

    void AsyncBenchmark_workload::calibration() {
        GET_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        is_cpu_calculations = true;
        is_gpu_calculations = false;
        is_omit_calculation_loop = false;
        int estcycles = p.get("calc_calibration").get_int("estimation_cycles");
        double timings[3];
        if (estcycles == 0) {
            throw std::runtime_error("AsyncBenchmark_workload: either -cycles_per_10usec or -estcycles option is required.");
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

    void AsyncBenchmark_workload::init() {
        AsyncBenchmark::init();
        GET_PARAMETER(std::vector<int>, calctime);
        GET_PARAMETER(std::vector<int>, len);
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

    void AsyncBenchmark_workload::finalize() {
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

    bool AsyncBenchmark_workload::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
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

    DECLARE_INHERITED(AsyncBenchmark_workload, workload)
    DECLARE_INHERITED(AsyncBenchmark_calibration, calc_calibration)
}
