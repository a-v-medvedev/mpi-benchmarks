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
#include <typeinfo>
#include <memory>

class any
{
    struct holder_base
    {
        virtual void *get() const { return NULL; }
        virtual const std::type_info &get_type_id() const { return typeid(void); }
        virtual ~holder_base() {}
        int dummy;
    };
    template <class type>
    struct holder : holder_base
    {
        std::shared_ptr<type> storedObject;
        holder(std::shared_ptr<type> pobject) : storedObject(pobject) {}
        virtual void *get() const { return storedObject.get(); }
        virtual const std::type_info &get_type_id() const { return typeid(type); }
    };
    std::shared_ptr<holder_base> held;
public:
    any() {}
    template <class type>
    any(std::shared_ptr<type> objectToStore) : held(new holder<type>(objectToStore))
  {}
    template <class type>
    type *as() const { 
        if (held.get() == NULL)
            return NULL;
        if (typeid(type) == held->get_type_id()) 
            return static_cast<type *>(held->get()); 
        else 
            return NULL;
    }
};

