# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _vta-get-started:

Get Started with VTA
====================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This is an introduction tutorial on how to use TVM to program the VTA design.

In this tutorial, we will demonstrate the basic TVM workflow to implement
a vector addition on the VTA design's vector ALU.
This process includes specific scheduling transformations necessary to lower
computation down to low-level accelerator operations.

To begin, we need to import TVM which is our deep learning optimizing compiler.
We also need to import the VTA python package which contains VTA specific
extensions for TVM to target the VTA design.
"""
from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np

env = vta.get_env()

from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator

host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))

if env.TARGET == "pynq" or env.TARGET == "de10nano":

    # Make sure that TVM was compiled with RPC=1
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)

    # Reconfigure the JIT runtime
    vta.reconfig_runtime(remote)

    # Program the FPGA with a pre-compiled VTA bitstream.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    vta.program_fpga(remote, bitstream=None)

# In simulation mode, host the RPC server locally.
elif env.TARGET in ("sim", "tsim", "intelfocl"):
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

m = 16
o = 1
A = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="A", dtype=env.acc_dtype)
B = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="B", dtype=env.acc_dtype)

# A copy buffer
A_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT), lambda *i: A(*i), "A_buf")
# B copy buffer
B_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT), lambda *i: B(*i), "B_buf")

# Describe the in-VTA vector addition
C_buf = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda *i: A_buf(*i).astype(env.acc_dtype) + B_buf(*i).astype(env.acc_dtype),
    name="C_buf",
)

# Cast to output type, and send to main memory
C = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
)

# Let's take a look at the generated schedule
s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# Set the intermediate tensors' scope to VTA's on-chip accumulator buffer
s[A_buf].set_scope(env.acc_scope)
s[B_buf].set_scope(env.acc_scope)
s[C_buf].set_scope(env.acc_scope)

# DMA transfer operation
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)

# Tell TVM that the computation needs to be performed
# on VTA's vector ALU
s[C_buf].pragma(C_buf.op.axis[0], env.alu)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# Let's take a look at the finalized schedule
print(vta.lower(s, [A, B, C], simple_mode=True))

my_vadd = vta.build(
    s, [A, B, C], tvm.target.Target("ext_dev", host=env.target_host), name="my_vadd"
)

# Write the compiled module into an object file.
temp = utils.tempdir()
my_vadd.save(temp.relpath("vadd.o"))

# Send the executable over RPC
remote.upload(temp.relpath("vadd.o"))

# Loading the Module
f = remote.load_module("vadd.o")

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the A and B arrays randomly in the int range of (-128, 128]
A_orig = np.random.randint(-128, 128, size=(o * env.BATCH, m * env.BLOCK_OUT)).astype(A.dtype)
B_orig = np.random.randint(-128, 128, size=(o * env.BATCH, m * env.BLOCK_OUT)).astype(B.dtype)

# Apply packing to the A and B arrays from a 2D to a 4D packed layout
A_packed = A_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
B_packed = B_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
A_nd = tvm.nd.array(A_packed, ctx)
B_nd = tvm.nd.array(B_packed, ctx)
C_nd = tvm.nd.array(np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(C.dtype), ctx)

# Invoke the module to perform the computation
f(A_nd, B_nd, C_nd)

# Compute reference result with numpy
C_ref = (A_orig.astype(env.acc_dtype) + B_orig.astype(env.acc_dtype)).astype(C.dtype)
C_ref = C_ref.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(C_ref, C_nd.numpy())
print("Successful vector add test!")
