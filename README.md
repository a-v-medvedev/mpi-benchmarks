# IMB-ASYNC Benchmark 
**based on: Intel(R) MPI Benchmarks 2019**

[![3-Clause BSD License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](license/license.txt)

## Introduction

The `IMB-ASYNC` benchmark suite is a collection of microbenchmark tools that help to fairly estimate the MPI asynchronous progress performance (computation-communication overlap) in many useful scenarios.

## Citation

The benchmark and the methodology is described in:
```
Alexey V. Medvedev "IMB-ASYNC: a revised method and benchmark to estimate MPI-3
asynchronous progress efficiency". Cluster Computing (2022) 25:2683–2697
```
DOI: [10.1007/s10586-021-03452-8](https://doi.org/10.1007/s10586-021-03452-8)\
Full text: [here](https://www.researchgate.net/publication/357865882_IMB-ASYNC_a_revised_method_and_benchmark_to_estimate_MPI-3_asynchronous_progress_efficiency#fullTextFileContent)

Please make a citation of this paper if you use this benchmark code in research.

## Build notes

The benchmark requires two small libraries for command line and config parsing. The download-and-build script for these libraries is placed in the `src/ASYNC/thirdparty` directory. It must download, build, and install the resulting files in the right place. The benchmark build code will link these libraries statically into the resulting benchmark binary.

So benchmark build is a two-step process:

- `cd src/ASYNC/thrirdparty; ./download-and-build.sh`
- `make CXX=<mpi-c++-wrapper> [WITH_CUDA=TRUE]`

Here the `<mpi-c++-wrapper>` denotes the actual MPI wrapper for C++ compiler. The default value for `CXX` is `mpicxx`.

## Benchmark groups

The individual benchmarks include:
- `sync_pt2p2`, `async_pt2pt` -- point-to-point benchmark where each rank exchanges with a predefined set of other ranks. Communications peers are defined by the topology, described [below](#topology-options). Synchronous variant utilizes `MPI_Send()`/`MPI_Recv()` function calls. The asynchronous variant uses an equivalent `MPI_Isend()`/`MPI_Irev()`/`MPI_Wait()` combination, and CPU/GPU calculation workload is optionally called before `MPI_Wait()` call to simulate communication/computation overlap. The calculation workload options are described [below](#calculation-workload-options).
- `sync_allreduce`, `async_allreduce` --  `MPI_Allreduce()` and `MPI_Iallreduce()`/`MPI_Wait()` benchmarks for the whole `MPI_COMM_WORLD` communicator or split subcommunicators, as it is defined by the topology. Calculation workload is optionally called before `MPI_Wait()` call.
- `sync_na2a`, `async_na2a` -- the same idea as in point-to-point benchmark where each rank exchanges with a predefined number of other ranks, is implemented using the neighborhood all-to-all collective operation. The topology is simply mapped to `MPI_Dist_graph_create_adjacent()`. The communication itself is implemented with a single `MPI_Neighbor_alltoall()` call for the synchronous variant and with `MPI_Ineighbor_alltoall()`/`MPI_Wait()` combination for the asynchronous one. Calculation workload is optionally called before `MPI_Wait()` call.
- `sync_rma_pt2pt`, `async_rma_pt2pt` -- the same idea as in point-to-point benchmark where each rank exchanges with a predefined number of other ranks, is implemented using the one-sided communication MPI functions. This is simply a one-sided communication version of the `sync_pt2pt`/`async_pt2pt` benchmark pair. Implemented with one-sided `MPI_Get()/MPI_Put()` pair in lock/unlock semantics; the `MPI_Rget()`/`MPI_Wait()` is used in an asynchronous variant. Calculation workload is optionally called before `MPI_Wait()`.

## Topology options

For each benchmark one can choose a topology (or a type of communication pattern) from a set. With command line options, one can chose the topology that makes sense for a particular benchmark. In fact, the current set of benchmarks is virtually divided into two parts: one implying point-to-point communication patterns (topologies), and another one implying collective communication patterns (topologies).

### Point-to-point style topologies

All of the communication topologies from the list below make sense for the benchmarks: `sync_pt2p2`, `async_pt2pt`, `sync_rma_pt2pt`, `async_rma_pt2pt`, `sync_na2a`, `async_na2a`.

- `ping-pong` -- is the topology of pairwise communications. All the ranks in `MPI_COMM_WORLD` are split into pairs, communicating with each other by sending the data block back and forth. The MPI functions that are used for this purpose depend on the specific benchmark. The parameters that tune this topology are:

    * `stride` -- an integer parameter, that defines the distance between pair elements. For example, `stride=1` means that the closest neighbor for the rank is going to be the counterpart for pairwise for communication; `stride=(MPI_COMM_WORLD size)/2` means that all ranks in `MPI_COMM_WORLD` are separated into two parts: first half of ranks is going to communicate the second half of ranks, preserving the rank order. The latter option is also meant by the specially handled value `stride=0` (it is the default one).
    * `bidirectional` -- `true`/`false`: defines options for bidirectional or unidirectional kind of pairwise communication (`true` is the default)

- `neighb` -- is the topology of exchange with two, four, or more closest neighbor ranks. The difference with the pairwise communication pattern, where each rank has always only one peer for communication, the exchanges in the `neighb` topology are arranged for even number of peers: the closest `N` neighbors that have greater rank numbers and smaller rank numbers than the rank in question make it a total `2*N` peers. The MPI functions that are used for this purpose depend on the specific benchmark. The parameters that tune this topology are:

    * `nneighb` -- the number of neighbors on each side to communicate with. For example:
        - for the `nneighb=2` parameter value, the rank with the number `N` is going to exchange data with ranks `N-1`, `N-2`, `N+1`, `N+2` (4 peers). 
        - for the `nneighb=3` parameter value, the rank with the number `N` is going to exchange data with ranks `N-1`, `N-2`, `N-3` `N+1`, `N+2`, `N+3` (6 peers).

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The defaut value for this parameter in `nneighb=1`.
    * `bidirectional` -- has the same meaning as for the `ping-pong` topology.

- `halo` -- is the topology of N-dimentional halo-exchange pattern. The number of peers for communication depends on the number of dimensions. For 1D exchange, the topology appears to be equivalent to the `neighb` topology with `nneighb=1`. For 2D, 3D, and 4D cases the number of peers is twice the number of dimensions, and the specific set of ranks to communicate is defined by linearization of the N-dimensional topology.  The MPI functions that are used for this purpose depend on the specific benchmark. The parameters that tune this topology are:

    * `ndim` -- the number of dimensions in the N-dimensional halo-exchange topology. The default value is `ndim=1`, and the allowed values are: `1, 2, 3, 4`.
    * `bidirectional` -- has the same meaning as for the `ping-pong` topology.

### Collective style topologies

The topologies listed below are meaningful for collective communication benchmarks: `sync_allreduce`, `async_allreduce`.

- `split` -- is the way to split the `MPI_COMM_WORLD` communicator into separate groups so that the collective communication happens within each of those sections independently. The collective communication function that is measured on this rank topology depends on the specific benchmark. The parameters that tune this topology are:

    * `nparts` -- an integer that defines the number of groups to split `MPI_COMM_WORLD` into. The default value is `nparts=1`, that is: no splitting, the whole `MPI_COMM_WORLD` is used for collective communication.
    * `combination` -- `separate`/`interleaved`: the way ranks are combined. For the `separate` option, the sequential ranks form groups; for the `interleaved` option, the groups are interleaving. For example:
        - 8 ranks that are split into 4 groups with `combination=split`. The correspondence of ranks and groups looks like this:\
          `{ rank=0: group=0; rank=1: group=0; rank=2: group=1; rank=3: group=1; rank=4: group=2; rank=5: group=2; rank=6: group=3; rank=7: group=3 }`. 
        - the same case but with `combination=interleaved`. The correspondence looks different :\
          `{ rank=0: group=0; rank=1: group=1; rank=2: group=2; rank=3: group=3; rank=4: group=0; rank=5: group=1; rank=6: group=2; rank=7: group=3 }`.
    * `nactive` -- an integer parameter, that applies only to the `combination=interleaved` case. It defines how many of the groups are going to be inactive, which means, simply skipping any communication. This is a useful parameter to define sparse collective communication topologies. For example: 
        * for 8 ranks split into 4 groups in the interleaved manner, we can define `nactive=1`, and get the topology:\
          `{ rank=0: group=0; rank=1: IDLE; rank=2: IDLE; rank=3: IDLE; rank=4: group 0; rank=5: IDLE; rank=6: IDLE; rank=7: IDLE }`. 
    
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The default value is `nactive=nparts`.

### Setting up the topology options

The topology and its parameters are defined for each benchmark separately. The command line (or equivalent YAML-file) option is: `-BENCH_params`, where `BENCH` stands for one of the short benchmark names: `pt2pt`, `na2a`, `rma_pt2pt`, `allreduce`. The parameter values for these options are the list of definitions in the form: `keyword=value:keyword=value:...`, where the keyword is either `topology` to denote the communication topology or the parameter names for a selected topology. For example:

* Option: `-pt2pt_params topology=ping-pong:stride=1:bidirectional=false` sets up the `ping-pong` pairwise topology for both `sync_pt2pt` and `async_pt2pt` benchmarks, and defines the parameters of it
* Option: `-allreduce_params topology=split:combination=separate:nparts=2` sets up the `separate` topology for both `sync_allreduce` and `async_allreduce` benchmarks.

## Calculation workload options

The option `-workload_params` controls the background calculation (CPU/GPU load) cycle parameters. The syntax for this option is similar to the `-BENCH_params` syntax, i.e. has the form: `keyword=value:keyword=value:...`. The keywords are:

* `cpu_calculations` -- `true`/`false`: to run or not the CPU load cycle in `async` versions of benchmarks. The CPU load cycle is the simplest small `DGEMM` kernel running in a loop. The number of cycles for the loops can be calibrated so that the runtime of the workload cycle can be tuned with 10 usec precision.
* `gpu_calcullations` -- `true`/`false`: to run on not the CUDA kernel with similar DGEMM calculations that may accompany the CPU cycle. Is meaningful only when the CPU load cycle is switched on. The time for this cycle is assumed to be close to the CPU load cycle time, but is kept with less precision and can last a little bit more than an expected time. It doesn't require a calibration.
* `cycles_per_10_usec` -- is a calibration parameter, that must be received by a previous calibration run.
* `omit_calc_over_est` -- `true`/`false`: can be used to omit the calculation slowdown impact estimation. Is useful when calculation slowdown in async communication modes is not expected, or we explicitly omit this estimation due to the low reliability of it on a specific system (due to uncontrolled CPU frequency changes or similar hardware effects).
* `manual_progress` -- `true`/`false`: in the CPU cycle, activate the regular `MPI_Test` function calls to facilitate `manual progress` mode.
* `spin_period` -- in usecs, set the `spin period`, i.e. time distance between sequential `MPI_Test` calls for `manual progress` mode.

## CPU load loop calibration

To precisely estimate the calculation slowdown effects of asynchronous communication progress actions, we need the CPU load cycle calibration to be made in advance. That means, in a "clean" mode, without any artificial progress measures, we have to run the special pseudo-benchmark named `calc_calibration`. This can be made only once for each system we plan to make measurements, but we must make sure the results of calibration are reliable. Please note, that for many modern HPC systems, the CPU load calibration and calculation slowdown measurement is tricky since they utilize some forms of dynamic CPU clock control. Only with these options switched off one can get any reliable results from this form of benchmarking!

The calibration is done like this:

`IMB-ASYNC calc_calibration`

In a successful case, this execution will report the calibration integer constant `cycles_per_10usec`, which must be remembered and put in the `-workload_params` during the actual benchmarking.

## Other options

There is a set of options controlling the config file reading, general benchmarking parameters and some other high-level aspects. Please refer the list below for the short description.

* `-dump config.yaml` -- create a config file based on the set of command line options that are given. This helps to make a yaml-config file to simplify future use of similar benchmarking scenarios
* `-load config.yaml` -- load a config file. Additional command line options can override those given in the config file
* `-output output.yaml` -- write out the structured YAML file with the benchmarking results at the end of the execution
* `-list` -- show a list of available benchmarks in the suite
* `-thread_level single|funneled|serialized|multiple|nompinit` -- controls the way `MPI_Init()` is called
* `-input filename` -- instructs IMB-ASYNC to read the list of benchmark to run from a text file. One becnhmark name per line
* `-include benchmark[,benchmark,[...]` -- add the listed benchmarks to the execution set
* `-exclude benchmark[,benchmark,[...]` -- remove the listed benchmarks from the execution set
* `-len INT,INT,...` -- list of message sizes to execute (multiply this size to a datatype sizeof() to get the message size in bytes
* `-datatype double|float|int|char` -- the MPI datatype that is used in all MPI exchange calls
* `-ncycles INT` -- the number of benchmark cycles
* `-nwarmup INT` -- the number of warmup cycles [default: 0]]
* `-calctime INT,INT,...` -- for each message size from `-len` option, set the calculation workload runtime in usecs.

## GPU aware mode

There is a mode in `IMB-ASYNC` suite which implies MPI operations over GPU memory. Only CUDA interface is implemented. To switch on this mode, one must rebuild the benchmark adding `WITH_CUDA=TRUE` command line argument to make.

(*FIXME* Additional options to be described...)

## Output YAML structure

The output YAML file is written out when the `-output` command line option is set. The file contains some collection of data for each benchmark that was executed. The structure can be outlined as follows:

```
<BENCH_NAME>:
    {tagv,tmin,tmax,over_full,over_comm,over_calc}:
        <message_len1>: <value_in_seconds>
        <message_len2>: <value_in_seconds>
        ...
        <message_lenN>: <value_in_seconds>
    topo:
        np: <NP>
        name: <TOPOLOGY_NAME>
        <rank0>: [ rank_to_comm1, rank_to_comm2, ..., rank_to_commN ]
        <rank1>: [ rank_to_comm1, rank_to_comm2, ..., rank_to_commN ]
        ...
        <rankN>: [ rank_to_comm1, rank_to_comm2, ..., rank_to_commN ]
<BENCH_NAME>:
    ...
```

In the `tagv`, `tmin`, `tmax`, `over_full`, `over_comm`, `over_calc` lists, for each tested messsage length, the values in seconds of the corresponding value is given. The `topo` list shows the name and the world communicator size for the topology that was used. For each rank the peers are enumerated for each communication event. That means, for point-to-point style of topology, "send" and "receive" communication events are enumerated separetely, and they all will be duplicated if the bidirectional type if communications is seleted.

For example, the benchmark line:

```
mpirun -np 4 ./IMB-ASYNC sync_pt2pt -pt2pt_params topology=halo:ndim=2 -len 4 -output out.yaml
```

produces the output YAML similar to:

```
sync_pt2pt: 
    tavg: 
        4: 2.4306409999999997e-06 
    tmin: 
        4: 2.289561e-06 
    tmax: 
        4: 2.4306780000000001e-06 
    over_full: 
        4: 0 
    over_comm: 
        4: 0 
    over_calc: 
        4: 0 
    topo:
        np: 4
        name: halo 
        0: [2, 2, 1, 1, 2, 2, 1, 1] 
        1: [3, 3, 0, 0, 3, 3, 0, 0] 
        2: [0, 0, 3, 3, 0, 0, 3, 3] 
        3: [1, 1, 2, 2, 1, 1, 2, 2]
```

Another example for colective benchmark:

```
mpirun -np 8 ./IMB-ASYNC sync_allreduce -allreduce_params topology=split:combination=interleaved:nparts=4:nactive=2 -len 4 -output out.yaml
```

Results in the `out.yaml` file with the contents:

```
sync_allreduce: 
    tavg: 
        4: 4.5656099999999172e-07
    tmin: 
        4: 0
    tmax:
        4: 3.9459020000000053e-06
    over_full: 
        4: 0
    over_comm: 
        4: 0
    over_calc: 
        4: 0
    topo: 
        np: 8
        name: split 
        0: [4] 
        1: [5] 
        2: [] 
        3: [] 
        4: [0] 
        5: [1] 
        6: [] 
        7: []
```

## Extending the suite and adding custom benchmarks

The `IMB-ASYNC` suite is designed to be extensible. One may add his own benchmarks and even new suites to the benchmarking engine. You can get the idea on how to do this by referring the special Intel(R) MPI Benchmark documentation [section](https://www.intel.com/content/www/us/en/developer/articles/technical/creating-custom-benchmarks-for-imb-2019.html). The `src/example` subdirectory contains the source code, described in this documentation piece.

## Copyright and Licenses

This benchmark suite inherits 3-Clause BSD License terms of Intel(R) MPI Benchmarks project, which it is based on.


(C) Intel Corporation (2016-2018)

(C) Alexey V. Medvedev (2019-2023)


