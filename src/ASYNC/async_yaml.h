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

struct YamlOutputMaker {
	std::string block;
	YamlOutputMaker(const std::string &_block) : block(_block) {}
	std::map<const std::string, double> kv;
	std::map<const std::string, std::vector<double>> kv_vec;
	std::map<const std::string, std::string> kv_str;
	void add(const std::string &key, const std::string &value) { kv_str[key] = value; }
	void add(const std::string &key, double value) { kv[key] = value; }
	void add(int key, double value) { add(std::to_string(key), value); }
    void add(const std::string &key, const std::vector<double> &values) { kv_vec[key] = values; }
	void add(int key, const std::vector<double> &values) { add(std::to_string(key), values); }
	void make_output(YAML::Emitter &yaml_out) const {
		yaml_out << YAML::Key << block << YAML::Value;
		yaml_out << YAML::Flow << YAML::BeginMap;
		for (auto &item : kv) {
			yaml_out << YAML::Key << YAML::Flow << item.first << YAML::Value << item.second;
		}
        for (auto &item : kv_str) {
			yaml_out << YAML::Key << YAML::Flow << item.first << YAML::Value << item.second;
		}
        for (auto &item : kv_vec) {
			yaml_out << YAML::Key << YAML::Flow << item.first << YAML::Value;
            yaml_out << YAML::BeginSeq;
            for (auto &elem : item.second) {
                yaml_out << elem;
            }
            yaml_out << YAML::EndSeq;
		}
		yaml_out << YAML::Flow << YAML::EndMap;
	}
};

static void WriteOutYaml(YAML::Emitter &yaml_out, const std::string &bname,
						 const std::vector<YamlOutputMaker> &makers) {
	yaml_out << YAML::Key << YAML::Flow << bname << YAML::Value;
	yaml_out << YAML::Flow << YAML::BeginMap;
	for (auto &m : makers) {
		m.make_output(yaml_out);
	}
	yaml_out << YAML::Flow << YAML::EndMap;
}


