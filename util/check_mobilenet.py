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
Deploy Pretrained Vision Model from MxNet on VTA
================================================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This tutorial provides an end-to-end demo, on how to run ImageNet classification
inference onto the VTA accelerator design to perform ImageNet classification tasks.
It showcases Relay as a front end compiler that can perform quantization (VTA
only supports int8/32 inference) as well as graph packing (in order to enable
tensorization in the core) to massage the compute graph for the hardware target.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user mxnet requests "Pillow<7"
#
# Now return to the python code. Import packages.
from __future__ import absolute_import, print_function

import argparse, json, os, requests, sys, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

import mxnet as mx
from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt
from mobilenetG import load_mobilenet
from collections import namedtuple

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform
from tvm.contrib.debugger import debug_executor

import vta
from vta.testing import simulator
from vta.top import graph_pack


# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

######################################################################
# Define the platform and model targets
# -------------------------------------
# Execute on CPU vs. VTA, and define the model.

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
#device = "vta"
device = "arm_cpu"

target = env.target if device == "vta" else env.target_vta_cpu
#target = env.target_vta_cpu

# Dictionary lookup for when to start/end bit packing
pack_dict = {
    "resnet18_v1": ["nn.conv2d", "nn.global_avg_pool2d"],
    "resnet18_v1-conv": ["nn.conv2d", "nn.global_avg_pool2d", 15,17],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "mobilenetv2_1.0": ["nn.conv2d", "nn.global_avg_pool2d",0,558],
    "mobilenetG" : ["nn.conv2d","nn.global_avg_pool2d",0,402],#217],
    "mobilenetGv2" : ["nn.conv2d","nn.global_avg_pool2d",0,680]

}

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
model = "resnet18_v1"#"mobilenetG"
assert model in pack_dict

######################################################################
# Obtain an execution remote
# --------------------------
# When target is 'pynq', reconfigure FPGA and runtime.
# Otherwise, if target is 'sim', execute locally.

if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "10.201.135.166")
    tracker_port = os.environ.get("TVM_TRACKER_PORT", 8103)
    
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
    device_host = os.environ.get("VTA_RPC_HOST", "115.145.209.238")
    device_port = os.environ.get("VTA_RPC_PORT", "3030")
    
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    
    else:
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
#ctx = remote.cpu(0)

######################################################################
# Build the inference graph executor
# ----------------------------------
# Grab vision model from Gluon model zoo and compile with Relay.
# The compilation steps are:
#
# 1. Front end translation from MxNet into Relay module.
# 2. Apply 8-bit quantization: here we skip the first conv layer,
#    and dense layer which will both be executed in fp32 on the CPU.
# 3. Perform graph packing to alter the data layout for tensorization.
# 4. Perform constant folding to reduce number of operators (e.g. eliminate batch norm multiply).
# 5. Perform relay build to object file.
# 6. Load the object file onto remote (FPGA device).
# 7. Generate graph executor, `m`.
#

# Load pre-configured AutoTVM schedules
with autotvm.tophub.context(target):
    # Populate the shape and data type dictionary for ImageNet classifier input
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    #mobilenet = load_mobilenet(pretrained=True)
    gluon_model = vision.get_model(model, pretrained=True)
    # Measure build start time
    build_start = time.time()

    # Start front end compilation
    #mod, params = relay.frontend.from_mxnet(mobilenet, shape_dict)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)
    #print(mod.astext(show_meta_data=False))

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    if target.device_name == "vta":
        # Perform quantization in Relay
        # Note: We set opt_level to 3 in order to fold batch norm
        with tvm.transform.PassContext(opt_level=2):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)
            # Perform graph packing and constant folding for VTA target
            assert env.BLOCK_IN == env.BLOCK_OUT
            # do device annotation if target is intelfocl or sim
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=pack_dict[model][0],
                stop_name=pack_dict[model][1],
                #start_name_idx=pack_dict[model][2],
                #stop_name_idx=pack_dict[model][3],
            )
    else:
        relay_prog = mod["main"]
    
    # Compile Relay program with AlterOpLayout disabled
    if target.device_name != "vta":
        with tvm.transform.PassContext(opt_level=2, disabled_pass={"AlterOpLayout","FoldScaleAxis"}):
            graph, lib, params = relay.build(
                relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
            )

    else:
        if env.TARGET == "intelfocl":
            # multiple targets to run both on cpu and vta
            target = {"cpu": env.target_vta_cpu, "ext_dev": target}
        with vta.build_config(
            opt_level=2, disabled_pass={"AlterOpLayout"}
        ):
            graph, lib, params = relay.build(
                relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
            )

    # Measure Relay build time
    build_time = time.time() - build_start
    #print(model + " inference graph built in {0:.2f}s!".format(build_time))

    # Send the inference library over to the remote RPC server
    
    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib.tar"))
    remote.upload(temp.relpath("graphlib.tar"))
    lib = remote.load_module("graphlib.tar")

    if env.TARGET == "intelfocl":
        ctxes = [remote.ext_dev(0), remote.cpu(0)]
        m = graph_executor.create(graph, lib, ctxes)
    else:
        # Graph runtime
        #m = graph_executor.create(graph, lib, ctx)

        m = debug_executor.create(graph, lib, ctx)

######################################################################
# Perform image classification inference
# --------------------------------------
# We run classification on an image sample from ImageNet
# We just need to download the categories files, `synset.txt`
# and an input test image.

# Download ImageNet categories
categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

# Download test image
image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
image_fn = "cat.png"
download.download(image_url, image_fn)

# Prepare test image for inference
image = Image.open(image_fn).resize((224, 224))
#plt.imshow(image)
#plt.show()
image = np.array(image) - np.array([123.0, 117.0, 104.0])
image /= np.array([58.395, 57.12, 57.375])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]

# Set the network parameters and inputs
m.set_input(**params)
m.set_input("data", image)

# Perform inference and gather execution statistics
# More on: :py:method:`tvm.runtime.Module.time_evaluator`
#num = 4  # number of times we run module for a single measurement
#rep = 3  # number of measurements (we derive std dev from this)

#print(">>>Profile start<<<")
record = m.profile()
print(record)
"""
print("=====TVM======")
m.set_input(**params)
m.set_input("data", image)
output = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0))).asnumpy()
output = np.argsort(output)[::-1]
print(output[0][0:10])
"""
"""
print("=====NET======")
for i in range(0,10):
    output = mobilenet(mx.nd.array(image)).asnumpy()
    output = np.argsort(output)[::-1]
    print(output[0][0:10])
"""
