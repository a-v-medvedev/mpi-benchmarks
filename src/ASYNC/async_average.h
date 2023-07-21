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

static inline double get_avg(double x, int nexec, int rank, int np, bool is_done) {
	double xx = x;
	std::vector<double> fromall;
	if (rank == 0)
		fromall.resize(np);
	if (!is_done) 
		xx = 0;
	MPI_Gather(&xx, 1, MPI_DOUBLE, fromall.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank != 0)
		return 0;
	const char *avg_option = nullptr;
	if (!(avg_option = getenv("IMB_ASYNC_AVG_OPT"))) {
		avg_option = "MEDIAN";
	}
	if (std::string(avg_option) == "MEDIAN") {
		std::sort(fromall.begin(), fromall.end());
		if (nexec == 0)
			return 0;
		int off = np - nexec;
		if (nexec == 1)
			return fromall[off];
		if (nexec == 2) {
			return (fromall[off] + fromall[off+1]) / 2.0;
		}
		return fromall[off + nexec / 2];
	}
	if (std::string(avg_option) == "AVERAGE") {
		double sum = 0;
		for (auto x : fromall)
			sum += x;
		sum /= fromall.size();
		return sum;
	}
	if (std::string(avg_option) == "MAX") {
		double maxx = 0;
		for (auto x : fromall)
			maxx = std::max(x, maxx);
		return maxx;
	}
	return -1;
}

