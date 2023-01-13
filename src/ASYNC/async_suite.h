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

#include <mpi.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>

#include "benchmark.h"
#include "benchmark_suites_collection.h"
#include "scope.h"
#include "utils.h"

#include "async_params.h"
#include "async_sys.h"
#include "async_alloc.h"
#include "extensions/params/params.inl"

namespace async_suite {

    enum gpu_mode_t { OFF, EXPLICIT, CUDAAWARE };
    enum gpu_select_t { HWLOC, COREMAP, GENERIC };
    static inline std::vector<std::string> remove_sync_tag(const std::vector<std::string> &benchs) {
        std::vector<std::string> result;
        std::set<std::string> s;
        for (const auto &b : benchs) {
            std::string bfiltered;
            if (b.substr(0, 5) == "sync_")
                bfiltered = b.substr(5, b.size() - 5);
            else if (b.substr(0, 6) == "async_")
                bfiltered = b.substr(6, b.size() - 6);
            else
                bfiltered = b;
            s.insert(bfiltered);
        }
        result.assign(s.begin(), s.end());
        return result;
    }

    #include "benchmark_suite.h"

    DECLARE_BENCHMARK_SUITE_STUFF(BS_GENERIC, async_suite)


    template <> bool BenchmarkSuite<BS_GENERIC>::declare_args(args_parser &parser,
                                                              std::ostream &output) const {
        UNUSED(output);
        parser.set_current_group(get_name());
        parser.add_vector<int>("len", "4,128,2048,32768,524288").
                     set_mode(args_parser::option::APPLY_DEFAULTS_ONLY_WHEN_MISSING).
                     set_caption("INT,INT,...");
        parser.add<std::string>("datatype", "double").
                     set_caption("double|float|int|char");
        parser.add_vector<int>("ncycles", "1000");
        parser.add<int>("nwarmup", 0).
                     set_caption("INT -- number of warmup cycles [default: 0]");
        parser.add_vector<int>("calctime", "10,10,50,500,10000").
                     set_mode(args_parser::option::APPLY_DEFAULTS_ONLY_WHEN_MISSING).
                     set_caption("INT,INT,...");
        parser.add<std::string>("gpumode", "off").
                     set_caption("off|explicit|cudaaware");
        parser.add<std::string>("hostallocmode", "stdc").
                     set_caption("stdc|mpi|cuda");
#ifdef WITH_HWLOC
        parser.add<std::string>("gpuselect", "generic").
                     set_caption("coremap|hwloc|generic");
#else
        parser.add<std::string>("gpuselect", "generic").
                     set_caption("coremap|generic");
#endif
        parser.add<std::string>("coretogpu", "").
                     set_caption("- core to GPU devices map, like: 0,1,2,3@0;4,5,6,7@1");


        for (const auto &b : remove_sync_tag(get_instance().names_list)) {
            std::string list_name = std::string(b) + "_params";
            parser.add_map(list_name.c_str(), "");
        }
        parser.set_default_current_group();
        return true;
    }

    std::vector<int> len;
    std::vector<int> calctime;
    MPI_Datatype datatype;
    YAML::Emitter yaml_out;
    std::string yaml_outfile;
    std::vector<int> ncycles;
    int nwarmup;
	params::dictionary<params::benchmarks_params> p;
    std::string coretogpu;
    gpu_mode_t gpu_mode;
    sys::host_alloc_t host_alloc_mode; 
    gpu_select_t gpu_selection_mode; 

    template <> bool BenchmarkSuite<BS_GENERIC>::prepare(const args_parser &parser,
                                                         const std::vector<std::string> &benchmarks_to_run,
                                                         const std::vector<std::string> &unknown_args,
                                                         std::ostream &output) {
        if (unknown_args.size() != 0) {
            output << "Some unknown options or extra arguments. Use -help for help." << std::endl;
            return false;
        }
        auto components = remove_sync_tag(benchmarks_to_run);
        if (std::find(components.begin(), components.end(), "calc_calibration") != components.end()) {
            if (components.size() != 1) {
                throw std::runtime_error("'calc_calibration' cannot be combined with any other benchmark");
            }
        } else {
            components.push_back("workload");
        }
		for (const auto &c : components) {
			std::string list_name = std::string(c) + "_params";
			std::map<std::string, std::string> pmap;
			parser.get(list_name.c_str(), pmap);
			p.add(c, { "component_details:", c });
			auto &list = p.get(c);
			for (auto &m : pmap) {
				list.parse_and_set_value(m.first, m.second);
			}
		}
        p.set_defaults();
        params::benchmarks_params::set_output(output);
        p.print();

        parser.get<int>("len", len);
        parser.get<int>("calctime", calctime);

        std::string dt = parser.get<std::string>("datatype");
        if (dt == "int") 
            datatype = MPI_INT;
        else if (dt == "double") 
            datatype = MPI_DOUBLE;
        else if (dt == "float") 
            datatype = MPI_FLOAT;
        else if (dt == "char") 
            datatype = MPI_CHAR;
        else {
            output << get_name() << ": " << "Unknown data type in 'datatype' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
        parser.get<int>("ncycles", ncycles);
        nwarmup = parser.get<int>("nwarmup");

        auto gm = parser.get<std::string>("gpumode");
        if (gm == "off") 
            gpu_mode = gpu_mode_t::OFF;
        else if (gm == "explicit")
            gpu_mode = gpu_mode_t::EXPLICIT;
        else if (gm == "cudaaware")
            gpu_mode = gpu_mode_t::CUDAAWARE;
        else {
            output << get_name() << ": " << "Unknown GPU mode in 'gpumode' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
//                     set_caption("off|explicit|cudaaware");
        auto ham = parser.get<std::string>("hostallocmode");
        if (ham == "stdc") 
            host_alloc_mode = sys::host_alloc_t::HA_STDC;
        else if (ham == "mpi")
            host_alloc_mode = sys::host_alloc_t::HA_MPI;
        else if (ham == "cuda")
            host_alloc_mode = sys::host_alloc_t::HA_CUDA;
        else {
            output << get_name() << ": " << "Unknown host memory allocation mode in 'hostallocmode' option."
                                            " Use -help for help." << std::endl;
            return false;
        }

        //             set_caption("stdc|mpi|cuda");
//                     set_caption("cuda");
        auto gs = parser.get<std::string>("gpuselect");
        if (gs == "coremap") 
            gpu_selection_mode = gpu_select_t::COREMAP;
#ifdef WITH_HWLOC
        else if (gs == "hwloc")
            gpu_selection_mode = gpu_select_t::HWLOC;
#endif        
        else if (gs == "generic")
            gpu_selection_mode = gpu_select_t::GENERIC;
        else {
            output << get_name() << ": " << "Unknown GPU selection mode in 'gpuselect' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
//                     set_caption("coremap|hwloc|generic");
        coretogpu = parser.get<std::string>("coretogpu");
        if (coretogpu.empty() && gpu_selection_mode == gpu_select_t::COREMAP) {
            throw std::runtime_error("'coremap' GPU selection option requires setting up the 'coretogpu' argument.");
        }
        if (!coretogpu.empty() && gpu_selection_mode != gpu_select_t::COREMAP) {
            throw std::runtime_error("'coretogpu' argument is meaningful only for 'coremap' type of GPU selection.");
        }

#ifdef WITH_HWLOC
        if (gs == gpu_select_t::HWLOC) {
            if (!sys::gpu_conf_init_with_hwloc())
                return false;
        } else {
            if (!sys::gpu_conf_init(coretogpu)) {
                return false;
            }
        }
#else
        if (!sys::gpu_conf_init(coretogpu))
            return false;
#endif

        yaml_outfile = parser.get<std::string>("output");
        yaml_out << YAML::BeginDoc;
        yaml_out << YAML::BeginMap;
        return true;
     }

     template <> void BenchmarkSuite<BS_GENERIC>::finalize(const std::vector<std::string> &,
                          std::ostream &, int rank) {
        yaml_out << YAML::EndMap;
        yaml_out << YAML::Newline;
        if (!yaml_outfile.empty() && !rank) {
            std::ofstream ofs(yaml_outfile, std::ios_base::out | std::ios_base::trunc);
            ofs << yaml_out.c_str();
        }
    }

	template <class SUITE>
	bool is_not_default(const std::string &name) {
		std::shared_ptr<Benchmark> b = SUITE::get_instance().create(name);
		if (b.get() == nullptr) {
			return true;
		}
		return !b->is_default();
	}

    template <> void BenchmarkSuite<BS_GENERIC>::get_bench_list(std::vector<std::string> &benchs,
                                          BenchmarkSuiteBase::BenchListFilter filter) const {
		get_full_list(benchs);
		if (filter == BenchmarkSuiteBase::DEFAULT_BENCHMARKS) {
			for (size_t i = benchs.size() - 1; i != 0; i--) {
				if (is_not_default<BenchmarkSuite<BS_GENERIC>>(benchs[i]))
					benchs.erase(benchs.begin() + i);
			}
		}
	}

	template <> void BenchmarkSuite<BS_GENERIC>::get_bench_list(std::set<std::string> &benchs,
                                          BenchmarkSuiteBase::BenchListFilter filter) const {
		get_full_list(benchs);
		if (filter == BenchmarkSuiteBase::DEFAULT_BENCHMARKS) {
			for (auto &bench_name : benchs) {
				if (is_not_default<BenchmarkSuite<BS_GENERIC>>(bench_name))
					benchs.erase(bench_name);
			}
		}
	}
     

#define HANDLE_PARAMETER(TYPE, NAME) if (key == #NAME) { \
                                        result = std::shared_ptr< TYPE >(&NAME, []( TYPE *){}); \
                                     }

#define GET_PARAMETER(TYPE, NAME) TYPE *p_##NAME = suite->get_parameter(#NAME).as< TYPE >(); \
                                  assert(p_##NAME != NULL); \
                                  TYPE &NAME = *p_##NAME;

    template <> any BenchmarkSuite<BS_GENERIC>::get_parameter(const std::string &key) {
        any result;
        HANDLE_PARAMETER(std::vector<int>, len);
        HANDLE_PARAMETER(std::vector<int>, calctime);
        HANDLE_PARAMETER(MPI_Datatype, datatype);
        HANDLE_PARAMETER(YAML::Emitter, yaml_out);
        HANDLE_PARAMETER(std::vector<int>, ncycles);
        HANDLE_PARAMETER(int, nwarmup);
        HANDLE_PARAMETER(params::dictionary<params::benchmarks_params>, p);
        HANDLE_PARAMETER(gpu_mode_t, gpu_mode);
        HANDLE_PARAMETER(sys::host_alloc_t, host_alloc_mode); 
        HANDLE_PARAMETER(gpu_select_t, gpu_selection_mode); 
        return result;
    }

}
