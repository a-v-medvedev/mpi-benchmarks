/*****************************************************************************
 *                                                                           *
 * Copyright 2016-2018 Intel Corporation.                                    *
 *                                                                           *
 *****************************************************************************

This code is covered by the Community Source License (CPL), version
1.0 as published by IBM and reproduced in the file "license.txt" in the
"license" subdirectory. Redistribution in source and binary form, with
or without modification, is permitted ONLY within the regulations
contained in above mentioned license.

Use of the name and trademark "Intel(R) MPI Benchmarks" is allowed ONLY
within the regulations of the "License for Use of "Intel(R) MPI
Benchmarks" Name and Trademark" as reproduced in the file
"use-of-trademark-license.txt" in the "license" subdirectory.

THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT
LIMITATION, ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. Each Recipient is
solely responsible for determining the appropriateness of using and
distributing the Program and assumes all risks associated with its
exercise of rights under this Agreement, including but not limited to
the risks and costs of program errors, compliance with applicable
laws, damage to or loss of data, programs or equipment, and
unavailability or interruption of operations.

EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, NEITHER RECIPIENT NOR
ANY CONTRIBUTORS SHALL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING
WITHOUT LIMITATION LOST PROFITS), HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OR
DISTRIBUTION OF THE PROGRAM OR THE EXERCISE OF ANY RIGHTS GRANTED
HEREUNDER, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

EXPORT LAWS: THIS LICENSE ADDS NO RESTRICTIONS TO THE EXPORT LAWS OF
YOUR JURISDICTION. It is licensee's responsibility to comply with any
export regulations applicable in licensee's jurisdiction. Under
CURRENT U.S. export regulations this software is eligible for export
from the U.S. and can be downloaded by or otherwise exported or
reexported worldwide EXCEPT to U.S. embargoed destinations which
include Cuba, Iraq, Libya, North Korea, Iran, Syria, Sudan,
Afghanistan and any other country to which the U.S. has embargoed
goods and services.

 ***************************************************************************
*/

#include <mpi.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>

#include "benchmark.h"
#include "benchmark_suites_collection.h"
#include "scope.h"
#include "utils.h"
#include "argsparser.h"

namespace async_suite {

    #include "benchmark_suite.h"

    DECLARE_BENCHMARK_SUITE_STUFF(BS_GENERIC, async_suite)

    template <> bool BenchmarkSuite<BS_GENERIC>::declare_args(args_parser &parser,
                                                              std::ostream &output) const {
        UNUSED(output);
        parser.set_current_group(get_name());
        parser.add_vector<int>("len", "1,2,4,8").
                     set_mode(args_parser::option::APPLY_DEFAULTS_ONLY_WHEN_MISSING);
        parser.add<std::string>("datatype", "int").set_caption("int|char");
        parser.add<int>("ncycles", 1000);
        parser.set_default_current_group();
        return true;
    }

    std::vector<int> len;
    MPI_Datatype datatype;
    int ncycles;

    template <> bool BenchmarkSuite<BS_GENERIC>::prepare(const args_parser &parser,
                                                         const std::vector<std::string> &,
                                                         const std::vector<std::string> &unknown_args,
                                                         std::ostream &output) {
        if (unknown_args.size() != 0) {
            output << "Some unknown options or extra arguments." << std::endl;
            return false;
        }
        parser.get<int>("len", len);
        std::string dt = parser.get<std::string>("datatype");
        if (dt == "int") datatype = MPI_INT;
        else if (dt == "char") datatype = MPI_CHAR;
        else {
            output << get_name() << ": " << "Unknown data type in datatype option" << std::endl;
            return false;
        }
        ncycles = parser.get<int>("ncycles");
        return true;
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
        HANDLE_PARAMETER(MPI_Datatype, datatype);
        HANDLE_PARAMETER(int, ncycles);
        return result;
    }

}