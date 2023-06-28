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

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>

#include "argsparser.h"
#include "extensions/params/params.h"
#include "extensions/params/params.inl"

namespace params {

#define ALLFAMILIES {}                                                                                
#define NOMINMAX {}                                                                                  
#define ALLALLOWED {}                                                                                 
#define CHANGEABLE true
#define NONCHANGEABLE false

#define BEGIN_DETAILS_DICT(DETAILS, FAMILY_KEY) struct DETAILS { \
    static constexpr const char *dict_name = #DETAILS; \
    using my_dictionary = params::dictionary<DETAILS>; \
    using my_list = params::list<DETAILS>; \
    static std::string get_family_key() { return FAMILY_KEY; } \
    static std::string get_layer_prefix() { return "-"; } \
    static uint16_t get_nlayers() { return 1; } \
    static void print_stream(const std::stringstream &ss) { \
        std::stringstream my_ss(ss.str()); \
        std::string line; \
        while (std::getline(my_ss, line)) { \
            if (poutput) { \
                *poutput << line << std::endl; \
            } else { \
                std::cout << line << std::endl; \
            } \
        } \
    } \
    static void print_table(const my_dictionary &dict) { \
        size_t size = dict.size(); \
        for (size_t n = 0; n < size; n++) { \
            auto name = dict.get(n); \
            dict.print_list(name, "", true); \
        } \
    }

#define END_DETAILS() };

// FIXME Replace {"pt2pt", "allreduce", "rma_pt2pt", "na2a"} with {"!workload", "!calc_calibration"}?
//#define ONLYBENCHS {"pt2pt", "allreduce", "rma_pt2pt", "na2a"}

BEGIN_DETAILS_DICT(benchmarks_params, "component_details:")
    static std::ostream *poutput;
    static void set_output(std::ostream &output) { poutput = &output; }
    static const params::expected_params_t &get_expected_params() {
        using namespace params;
        static const std::vector<std::string> ONLYBENCHS = {"pt2pt", "allreduce", "rma_pt2pt", "na2a"};
        static const expected_params_t expected_params = {
            {"component_details:",  {value::S, NONCHANGEABLE, ALLFAMILIES,  NOMINMAX,   ALLALLOWED}},
            {"topology",            {value::S, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   ALLALLOWED}},
            {"combination",         {value::S, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   { "interleaved", "separate" }}},
            {"nparts",              {value::I, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   ALLALLOWED}},
            {"nactive",             {value::I, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   ALLALLOWED}},
            {"bidirectional",       {value::B, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   ALLALLOWED}},
            {"stride",              {value::I, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   ALLALLOWED}},
            {"nneighb",             {value::I, NONCHANGEABLE, ONLYBENCHS,   NOMINMAX,   ALLALLOWED}},
            {"ndim",                {value::I, NONCHANGEABLE, ONLYBENCHS,   {"1", "4"}, ALLALLOWED}},
            {"cpu_calculations",    {value::B, NONCHANGEABLE, {"workload"}, NOMINMAX,   ALLALLOWED}},
            {"gpu_calculations",    {value::B, NONCHANGEABLE, {"workload"}, NOMINMAX,   ALLALLOWED}},
            {"manual_progress",     {value::B, NONCHANGEABLE, {"workload"}, NOMINMAX,   ALLALLOWED}},
            {"spin_period",         {value::I, NONCHANGEABLE, {"workload"}, NOMINMAX,   ALLALLOWED}},
            {"cycles_per_10usec",   {value::I, NONCHANGEABLE, {"workload"}, NOMINMAX,   ALLALLOWED}},
            {"omit_calc_over_est",  {value::B, NONCHANGEABLE, {"workload"}, NOMINMAX,   ALLALLOWED}},
            {"estimation_cycles",   {value::I, NONCHANGEABLE, {"calc_calibration"}, NOMINMAX, ALLALLOWED}},
        };
        return expected_params;
    }
    static void set_family_defaults(my_list &list, const std::string &family,
                                    const std::string &list_name) {
        (void)family;
        if (list_name == "pt2pt") {
            list.set_value_if_missing<std::string>("topology", "ping-pong");
        }
        if (list_name == "allreduce") {
            list.set_value_if_missing<std::string>("topology", "split");
        }
        if (list_name == "rma_pt2pt") {
            list.set_value_if_missing<std::string>("topology", "ping-pong");
        }
        if (list_name == "na2a") {
            list.set_value_if_missing<std::string>("topology", "ping-pong");
        }
        if (list_name == "workload") {
            list.set_value_if_missing<bool>("cpu_calculations", false);
            list.set_value_if_missing<bool>("gpu_calculations", false);
            list.set_value_if_missing<bool>("manual_progress", false);
            list.set_value_if_missing<bool>("omit_calc_over_est", false);
            if (list.get_bool("cpu_calculations")) {
                if (list.get_bool("manual_progress")) {
                    list.set_value_if_missing<uint32_t>("spin_period", 50);
                }
                list.set_value_if_missing<uint32_t>("cycles_per_10usec", 0);
                if (list.get_int("cycles_per_10usec") == 0) {
					throw std::runtime_error("params: for 'cpu_calculations' workload: calibration parameter"
                                             " 'cycles_per_10usec' is obligatory. Run calc_calibration first!");
                }
            }
        }
        if (list_name == "calc_calibration") {
            list.set_value_if_missing<uint32_t>("estimation_cycles", 3);
        }
        if (list_name != "calc_calibration" && list_name != "workload") {
            if (list.get_string("topology") == "ping-pong") {
                list.set_value_if_missing<bool>("bidirectional", true);
                list.set_value_if_missing<uint32_t>("stride", 0);
            }
            if (list.get_string("topology") == "neighb") {
                list.set_value_if_missing<bool>("bidirectional", true);
                list.set_value_if_missing<uint32_t>("nneighb", 1);
            }
            if (list.get_string("topology") == "halo") {
                list.set_value_if_missing<bool>("bidirectional", true);
                list.set_value_if_missing<uint32_t>("ndim", 1);
            }
            if (list.get_string("topology") == "split") {
                list.set_value_if_missing<std::string>("combination", "interleaved");
                list.set_value_if_missing<uint32_t>("nparts", 1);
                if (list.get_string("combination") == "separate" && list.is_value_set("nactive")) {
                    throw std::runtime_error("params: for 'split' topology: parameter 'nactive'"
                                             " is not meaningful for 'separate' ranks combination");
                }
                list.set_value_if_missing<uint32_t>("nactive", list.get_int("nparts"));
            }
        }
    }
    static void set_dictionary_defaults(my_dictionary &dict) {
        (void)dict;
    }
END_DETAILS()

std::ostream *benchmarks_params::poutput = nullptr;

}



