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

all: announce

ANNOUNCE=0

ifdef TARGET
include $(TARGET)/Makefile.*.mk
BINARY:=IMB-$(TARGET)
else
include */Makefile.*.mk
BINARY:=imb
endif

ifeq ($(origin CC),default)
CC=mpicc
endif
ifeq ($(origin CXX),default)
CXX=mpicxx
endif
WITH_CUDA ?= FALSE
NV_CXX ?= nvcc
override CPPFLAGS += -I.


#override CFLAGS += -O2 -Wall -Wno-long-long
#override CXXFLAGS += -O2 -Wall -Wextra -pedantic -Wno-long-long -std=c++11

override CFLAGS += -O2 -g -Wall -Wno-long-long
override CXXFLAGS += -O2 -g -Wall -Wextra -pedantic -Wno-long-long -std=c++11

override LDFLAGS += -largsparser -lyaml-cpp 
ifdef WITH_OPENMP
override CFLAGS += -fopenmp
override CXXFLAGS += -fopenmp
endif

#ifeq ($(origin CC),default)
#CXX=mpicc
#endif
#ifeq ($(origin CXX),default)
#CC=mpicxx
#endif
#override CPPFLAGS += -I.
#override CFLAGS = -g -Wall -Wextra -pedantic -Wno-long-long
#CXXFLAGS = -g -Wall -Wextra -pedantic -Wno-long-long
#ifdef WITH_OPENMP
#override CFLAGS += -qopenmp
#override CXXFLAGS += -qopenmp
#endif



all: $(BINARY)

IMB_SRC = imb.cpp scope.cpp
IMB_OBJ = $(IMB_SRC:.cpp=.o) 

BECHMARK_SUITE_SRC += benchmark_suites_collection.cpp
BECHMARK_SUITE_OBJ1 = $(BECHMARK_SUITE_SRC:.cpp=.o)
BECHMARK_SUITE_OBJ = $(BECHMARK_SUITE_OBJ1:.cu=.o)

HEADERS += benchmark.h \
benchmark_suite_base.h \
benchmark_suite.h \
benchmark_suites_collection.h \
utils.h \
scope.h

announce:
	@if test "$(ANNOUNCE)" -eq "1"; then echo "NOTE: Building target: $(TARGET), binary name: $(BINARY)"; fi
	@if test "$(ANNOUNCE)" -eq "1"; then echo "NOTE: Use make TARGET=<DIR_NAME> to select a target suite"; fi

ifeq ($(WITH_CUDA),TRUE)
NVLDFLAGS=$(shell echo $(LDFLAGS) | sed 's/-Wl,/-Xlinker /g')
$(BINARY): $(IMB_OBJ) $(BECHMARK_SUITE_OBJ) $(ADDITIONAL_OBJ) 
	$(NV_CXX) -ccbin `which $(CXX)` $(CPPFLAGS) -o $@ $^ $(NVLDFLAGS)
else
$(BINARY): $(IMB_OBJ) $(BECHMARK_SUITE_OBJ) $(ADDITIONAL_OBJ) 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
endif	

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NV_CXX) $(CPPFLAGS) -O0 -g -c -o $@ $<

clean: announce
	rm -f $(IMB_OBJ) $(BECHMARK_SUITE_OBJ) $(ADDITIONAL_OBJ) $(BINARY) 


$(BECHMARK_SUITE_OBJ): $(HEADERS) Makefile
$(IMB_OBJ): $(HEADERS) Makefile

.PHONY: announce clean all

