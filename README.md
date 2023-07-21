# Intel(R) MPI Benchmarks: IMB-ASYNC
[![3-Clause BSD License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](license/license.txt)

--------------------------------------------------

The IMB-ASYNC benchmark suite is a collection of microbenchmark tools that
help to fairly estimate the MPI asynchronous progress performance (computation-communication overlap) 
in many useful scenarios.

The individual bechmarks include:
- sync_pt2p2, async_pt2pt -- ping-pong style point-to-point benchmark with stride between peers 
given with the  option "-stride". Synchronous variant utilizes MPI_Send()/MPI_Recv() function calls.
Asynchronous variant uses equivalent MPI_Isend()/MPI_Irev()/MPI_Wait() combination, and pure
calculation workload is optionally called before MPI_Wait() call (see the "-workload" option).
- sync_allreduce, async_allreduce -- MPI_Allreduce() and MPI_Iallreduce()/MPI_Wait() benchmarks for the
whole MPI_COMM_WORLD commuicator. Pure calculation workload is optionally called before MPI_Wait() call
(see the "-workload" option).
- sync_na2a, async_na2a -- messages exchnage with two closest neighbour ranks for each rank in 
MPI_COMM_WORLD. Implemented with MPI_Neighbor_alltoall() for synchronous variant and with 
MPI_Ineighbor_alltoall()/MPI_Wait() combination for the asynchronous one. Pure calculation workload 
is optionally called before MPI_Wait() call (see the "-workload" option).
- sync_rma_pt2pt, async_rma_pt2pt -- ping-pong stype message exchnage with a neighbour rank 
(respectig the "-stride" parameter). This is simply a one-sided communication version of
sync_pt2pt/async_pt2pt benchmark pair. Implemented with one-sided MPI_Get() call in 
lock/unlock semantics with MPI_Rget()/MPI_Wait() usage in an asynchronous variant. 
Pure calculation workload is optionally called before MPI_Wait() call (see the
"-workload" option).

The "calibration" pseudo-benchmark:
- calc_calibration -- is used to detect and report the calculation cycle calibration constant 
(to be used later as a "-cper10usec" parameter value).

The workload option (meaningful for asynchromous variants of all benchmarks):
- -workload none -- means: do nothing, just proceed with waiting the non-blocking operation 
to complete with MPI_Wait(). This is a default value for the "-workload" option;
- -workload calc -- spin a dummy calculation loop doing some repeating simple and small-scale dense 
linear algebra subroutine on each rank. The number of calculation cycles to run is determined from 
the constant given in "-calctime" parameter and the "-cper10usec" constant, also given as a parameter. 
The "-cper10usec" parameter value highly depends on the CPU model and speed, 
so it must be obtained beforehand with calc_calibration pseudo-benchmark on each particular machine.
The "-cper10usec" parameter is a required one for "calc" and "calc_progress" workload types.
- -workload calc_progress -- as in "calc" workload type, do some dummy calculation, but also
call MPI_Test() periodically. The "-spinperiod" parameter sets up how often to call MPI_Test().



----------------------
Copyright and Licenses
----------------------

This benchmark suite inherits 3-Clause BSD License terms of Intel(R) MPI Benchmarks project, 
which it is based on.


(C) Intel Corporation (2016-2018)

(C) Alexey V. Medvedev (2019-2023)
