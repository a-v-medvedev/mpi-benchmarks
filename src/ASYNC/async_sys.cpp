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

#include <assert.h>
#include "async_sys.h"

namespace sys {

struct gpu_conf {
    int ncores = 0, ngpus = 0;
    std::map<int, int> core_to_gpu;
    std::vector<int> cores;
    mutable int core = -1;
    int nthreads = 0;
    void init_generic(); // gets ncores and ngpus from system
    void init_from_str(const std::string &str);
    int gpu_by_core(int core) const {
        auto gpuit = core_to_gpu.find(core);
        assert(gpuit != core_to_gpu.end());
        return gpuit->second;
    }
};

void gpu_conf::init_generic() {
#ifdef WITH_CUDA    
    core_to_gpu.clear();
    size_t NC = getnumcores();
    size_t NG = cuda::get_num_of_devices();
    for (size_t i = 0; i < NC; i++) {
        int G = -1;
        if (NG)
            G = i * NG / NC;
        core_to_gpu[i] = G;
    }
    ncores = NC;
    ngpus = NG;
#endif    
    return;
}

static inline std::vector<std::string> str_split(const std::string &s, char delimiter)
{
   std::vector<std::string> result;
   std::string token;
   std::istringstream token_stream(s);
   while (std::getline(token_stream, token, delimiter)) {
      result.push_back(token);
   }
   return result;
}

static inline void vstr_to_vint(std::vector<std::string>& from,
                                std::vector<int>& to) {
    to.clear();
    for (auto& s : from) {
        int x = std::stoi(s);
        to.push_back(x);
    }
}

void gpu_conf::init_from_str(const std::string &str) {
    if (str.empty()) {
        init_generic();
        return;
    }
    try {
        auto s_numas = str_split(str, ';');
        size_t ngpus = 0;
        int numa = 0;
        for (auto& s_numa : s_numas) {
            std::vector<std::string> s_gpus;
            std::vector<int> gpus;
            auto s_core_gpu = str_split(s_numa, '@');
            if (s_core_gpu.size() == 2) {
                auto s_gpus = str_split(s_core_gpu[1], ',');
                vstr_to_vint(s_gpus, gpus);
                assert(s_gpus.size() == gpus.size());
            }
            std::vector<int> local_cores;
            auto s_cores = str_split(s_core_gpu[0], ',');
            if (s_cores.size() == 1 && (!strncasecmp(s_cores[0].c_str(), "0x", 2))) {
                uint64_t mask = 0;
                long long n = std::stoll(s_cores[0], nullptr, 16);
                assert(n > 0 && n < (long long)UINT64_MAX + 1);
                mask = (uint64_t)n;
                for (int j = 0; j < 64; j++) {
                    if ((uint64_t)mask & ((uint64_t)1 << j)) {
                        local_cores.push_back(j);
                    }
                }
            } else {
                vstr_to_vint(s_cores, local_cores);
                assert(s_cores.size() == local_cores.size());
            }
            size_t NC = local_cores.size();
            size_t NG = gpus.size();
            for (size_t i = 0; i < NC; i++) {
                int G = -1;
                if (NG)
                    G = gpus[i * NG / NC];
                core_to_gpu[local_cores[i]] = G;
            }
            numa++;
            ngpus += NG;
            ncores += NC;
            cores.insert(cores.end(), local_cores.begin(), local_cores.end());
        }
    } catch (std::runtime_error& ex) {
        std::cout << std::string("gpuconf: handling/parsing conf string failed,"
                                 " falling back to generic: ") + ex.what() << std::endl;
        init_generic();
        return;
    } catch (...) {
        std::cout << "gpuconf: handling/parsing conf string,"
                     " falling back to generic." << std::endl;
        init_generic();
        return;
    }
}

bool gpu_conf_init(const std::string &str)
{
#ifdef WITH_CUDA    
    gpu_conf conf;
    conf.init_from_str(str);
    if (cuda::get_num_of_devices() == 0) {
        std::cout << "FATAL: no GPU devices found" << std::endl;
        return false;
    }
    int nthreads = 0;
    if (!threadaffinityisset(nthreads)) {
        std::cout << "WARNING: thread affinity seems to be not set,"
                     " can't choose relevant GPU device" << std::endl;
        return true;
    }
    int gpu = conf.gpu_by_core(getthreadaffinity());
    assert(gpu != -1);
    cuda::set_current_device(gpu);
#endif    
    return true;
}


}


