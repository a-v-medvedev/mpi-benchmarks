#
# Copyright 2016-2018 Intel Corporation.                                    *
# Copyright 2019-2023 Alexey V. Medvedev
#
#   The 3-Clause BSD License
#
#   Copyright (C) Intel, Inc. All rights reserved.
#   Copyright (C) 2019-2023 Alexey V. Medvedev. All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#   3. Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#

override CPPFLAGS += -DASYNC
override CPPFLAGS += -IASYNC -D__USE_BSD

BECHMARK_SUITE_SRC += ASYNC/async_benchmark.cpp ASYNC/async_alloc.cpp ASYNC/async_sys.cpp
ifeq ($(WITH_CUDA),TRUE)
override CPPFLAGS += -DWITH_CUDA	
BECHMARK_SUITE_SRC += ASYNC/async_cuda.cu
endif

override CXXFLAGS += -IASYNC/thirdparty/include
override LDFLAGS += -LASYNC/thirdparty/lib -Wl,-rpath=ASYNC/thirdparty/lib -Wl,-rpath=. -lyaml-cpp -largsparser
