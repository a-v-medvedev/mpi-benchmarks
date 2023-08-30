# Intel(R) MPI Benchmarks: IMB-ASYNC
[![3-Clause BSD License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](license/license.txt)

## Introduction

The IMB-ASYNC benchmark suite is a collection of microbenchmark tools that help to fairly estimate the MPI asynchronous progress performance (computation-communication overlap) in many useful scenarios.

## Citation

The benchmak and the methodology is described in:
```
Alexey V. Medvedev "IMB-ASYNC: a revised method and benchmark to estimate MPI-3asynchronous progress efficiency". Cluster Computing (2022) 25:2683–2697
```
DOI: [10.1007/s10586-021-03452-8](https://doi.org/10.1007/s10586-021-03452-8)

Full text: [here](https://www.researchgate.net/publication/357865882_IMB-ASYNC_a_revised_method_and_benchmark_to_estimate_MPI-3_asynchronous_progress_efficiency#fullTextFileContent)

Please make a citation of this paper if you use this benchmark code in research.

## Build notes

The benchmark requires two small libraries for command line and config parsing. The download and build script for these libraries is placed in the `src/ASYNC/thirdparty` directory. It must download, build and place the resulting files in a right place. The benchmark build code will link these libraries statically into the resulting benchmark binary.

## Benchmark groups

The individual bechmarks include:
- `sync_pt2p2`, `async_pt2pt` -- ping-pong style point-to-point benchmark with some stride between peers that is defined by topology. Synchronous variant utilizes `MPI_Send()`/`MPI_Recv()` function calls.
Asynchronous variant uses equivalent `MPI_Isend()`/`MPI_Irev()`/`MPI_Wait()` combination, and pure
calculation workload is optionally called before `MPI_Wait()` call.
- `sync_allreduce`, `async_allreduce` -- `MPI_Allreduce()` and `MPI_Iallreduce()`/`MPI_Wait()` benchmarks for the whole `MPI_COMM_WORLD` commuicator. Pure calculation workload is optionally called before `MPI_Wait()` call.
- `sync_na2a`, `async_na2a` -- messages exchnage with two closest neighbour ranks for each rank in `MPI_COMM_WORLD`. Implemented with `MPI_Neighbor_alltoall()` for synchronous variant and with
`MPI_Ineighbor_alltoall()`/`MPI_Wait()` combination for the asynchronous one. Pure calculation workload is optionally called before `MPI_Wait()` call.
- `sync_rma_pt2pt`, `async_rma_pt2pt` -- ping-pong stype message exchnage with a neighbour rank. This is simply a one-sided communication version of `sync_pt2pt`/`async_pt2pt` benchmark pair. Implemented with one-sided `MPI_Get()` call in lock/unlock semantics with `MPI_Rget()`/`MPI_Wait()` usage in an asynchronous variant. Pure calculation workload is optionally called before `MPI_Wait()`.
## Topology options

For each benchmark one can use a specific set of possible communication topologies -- the ones that make sense for a particular benchmark. In fact, the current set of benchmarks is virtually divided into two parts: one implying point-to-point communication patterns, and another one implying collective communication patterns.

### Point-to-point style topologies

All of the communication topologies from the list below make sense for the benchmarks: `sync_pt2p2`, `async_pt2pt`, `sync_rma_pt2pt`, `async_rma_pt2pt`, `sync_na2a`, `async_na2a`.

- `ping-pong` -- is the topology of pairwise communications. All the ranks in `MPI_COMM_WORLD` are split into pairs, communicating with each other sending the data block back and forth. The MPI functions that are used for this purpose depend on the specific benchmark. The parameters that tune this topology are:

    * `stride` -- an integer parameter, that defines the distance between pair elements. For example, `stride=1` means that the closest neghbour for the rank is going to be the counterpart for pairwise for communication; `stride=(MPI_COMM_WORLD size)/2` means that all ranks in `MPI_COMM_WORLD` are seprated into two parts: first half of ranks is going to communicate the second half of ranks, preserving the rank order. The later option is also meant by the specially handled value `stride=0` (the defafult one).
    * `bidirectional` -- `true`/`false`: defines options for bidirectional or unidirectional kind of pairwise communication (`true` is default)

- `neighb` -- is the topology if exchange with two, four or more closest neighbour ranks. The difference with pairwise communication pattern, where each rank has always the only one peer for communication, the exchanges in `neighb` topology is arranged for power-of-two number of peers: the closest neighbour ranks that are greater and smaller than the rank in question. The MPI functions that are used for this purpose depend on the specific benchmark. The parameters that tune this topology are:

    * `nneighb` -- the number of neighbours on each side to communicate with. For example:
        - for `nneighb=2` parameter value, the rank with number `N` is going to exchange data with ranks `N-1`, `N-2`, `N+1`, `N+2`. 
        - for `nneighb=3` parameter value, the rank with number `N` is going to exchange data with ranks `N-1`, `N-2`, `N-3` `N+1`, `N+2`, `N+3`. 
    The defaut value for this parameter in `nneighb=1`.
    * `bidirectional` -- has the same meaning as for the `ping-pong` topology.

- `halo` -- is the topology of N-dimentional halo-exchange pattern. The number of peers for communication depends on the number of dimensions. For 1D exchange, the topology appears to be equivalent to `neighb` topology with `nneighb=1`. For 2D, 3D, 4D cases the number of peers is twice the number of dimensions, and the specific set of ranks to communicate is defined by linearization of the N-dimensional topology.  The MPI functions that are used for this purpose depend on the specific benchmark. The parameters that tune this topology are:

    * `ndim` -- the number of dimension in the N-dimensional halo-exchange topology. The default value in `ndim=1`, and the allowed values are: `1, 2, 3, 4`.
    * `bidirectional` -- has the same meaning as for the `ping-pong` topology.

### Collective style topologies

The topogies listed below are meaningful for collective communication benchmarks: `sync_allreduce`, `async_allreduce`.

- `split` -- is the way to split the `MPI_COMM_WORLD` communicator into separate groups, so that the collective communication happens withing each of those sections independently. The collective communication function that is measured on this rank topology depends on the specific benchmark. The parameters that tune this topology are:

    * `combination` -- `split`/`interleaved`: the way ranks are combined. For `split` option, the sequential ranks form groups; for `interleaved` option, the groups are interleaving. For example:
        - 8 ranks that are split into 4 groups with `combination=split`, the correspondance of ranks and groups looks like this: `{ rank=0: group=0; rank=1: group=0; rank=2: group=1; rank=3: group=1; rank=4: group=2; rank=5: group=2; rank=6: group=3; rank=7: group=3 }`. 
        - for the same case but with `combination=interleaved`, the correspondance looks differenly: `{ rank=0: group=0; rank=1: group=1; rank=2: group=2; rank=3: group=3; rank=4: group=0; rank=5: group=1; rank=6: group=2; rank=7: group=3 }`.
    * `nparts` -- an integer that defines the number of groups to split `MPI_COMM_WORLD` into. The default value is `nparts=1`, that is: no splitting, the whole `MPI_COMM_WORLD` is used for collective communication.
    * `nactive` -- an integer parameter, that is applicable only to `combination=interleaved` case. It defines how many of the groups are going to be inactive, that means, simply skipping any communication. This is a useful parameter to define sparse collective communication topologies. For example: 
        * for 8 ranks split into 2 groups in the interleaved manner, we can define `nactive=1`, and get the topology: `{ rank=0: group=0; rank=1: IDLE; rank=2: IDLE; rank=3: IDLE; rank=4: group 0; rank=5: IDLE; rank=6: IDLE; rank=7: IDLE }`. 
    The default value is `nactive=nparts`.

### Setting up the topology options

The topology and its parameters are defined for each benchmark separately. The command line (or equivalent YAML-file) option is: `-BENCH_params`, where `BENHCH` stands for one of short benchmark names: `pt2pt`, `na2a`, `rma_pt2pt`, `allreduce`. The parameter value for these options are list on definitions in the form: `keyword=value:keyword=value:...`, where keyword is either `topology` to denote the communication topology, or the parameter names for a selected topology. For example:

* Option: `-pt2pt_params topology=ping-pong:stride=1:bidirectional=false` sets up the `ping-pong` pairwise topology for both `sync_pt2pt` and `async_pt2pt` benchmarks, and defines the parameters of it
* Option: `-allreduce_params topology=split:combination=split:nparts=2` sets up the `split` topology for both `sync_allreduce` and `async_allreduce` bechmarks.

## Calculation workload options

The option `-workload_params` controls the background calculation load cycle parameters. The syntax for this option is similar to `-BENCH_params` syntax, i.e. has the form: `keyword=value:keyword=value:...`. The keywords are:

* `cpu_calculations` -- `true`/`false`: to run or not the CPU load cycle in `async` versions of benchmarks. The CPU load cycle is a simplest small DGEMM kernel running is a loop. The number of cycles for the loops can be calibrated so that the runtime of the workload cycle could be tuned with 10 usec precision.
* `gpu_calcullations` -- `true`/`false`: to run on not the CUDA kernel with similar DGEMM calculations that may accompany the CPU cycle. Is meaningful only when CPU load cycle is switched on. The time for this cycle is assumed to be close the CPU load cycle time, but is kept with less precision and can be a bit more lenghthy. Doesn't require a calibration.
* `cycles_per_10_usec` -- is a calibration parameter, that must be received by previos calibration run.
* `omit_calc_over_est` -- `true`/`false`: can be used to omit the calculation slowdown impact estimation. Is useful when calculation slowdown in async comunication modes is not expected, or we explicitely omit this estimation due to low realibility of it on a specific system (due to uncontrolled CPU frequency changes or similar hardware effects).
* `manual_progress` -- `true`/`false`: in CPU cycle, activate the regular `MPI_Test` function calls to facilitate `manual progress` mode.
* `spin_period` -- in usecs, set the `spin period`, i.e. time distance between sequential `MPI_Test` calls for `manual progress` mode.

## CPU load loop calibration

To precisely estimate the calculation slowdown effects of asynchronios communication progress actions, we need the CPU load cycle calibration to be made in advance. That means, in a "clean" mode, without any artificial progress measures, we have to run the special pseudo-benchmark named `calc_calibration`. This can be made only once for each system we plan to make measurements, but we must make sure the results of calibration are reliable. Please note, thet for many modern HPC systems the CPU load calibration is tricky since they itilize some forms of dynamic CPU clock control. Only with these options switched off one can get any leliable results from this form of benchmarking!

The calibration is done like this:

`IMB-ASYNC calc_calibration`

In a successfull case, this execution will report the calibration integer constant `cycles_per_10usec`, that must be remebered and put in the `-workload_params` during the actual benchmarking.

## Copyright and Licenses

This benchmark suite inherits 3-Clause BSD License terms of Intel(R) MPI Benchmarks project, which it is based on.


(C) Intel Corporation (2016-2018)

(C) Alexey V. Medvedev (2019-2023)


