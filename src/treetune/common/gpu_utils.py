#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Outputs some information on CUDA-enabled devices on your computer,
including current memory usage.
It's a port of https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
from C to Python with ctypes, so it can run without compiling anything. Note
that this is a direct translation with no attempt to make the code Pythonic.
It's meant as a general demonstration on how to obtain CUDA device information
from Python without resorting to nvidia-smi or a compiled Python extension.
Author: Jan Schlüter
"""

import ctypes
import os
import subprocess
import time

from treetune import logging_utils

logger = logging_utils.get_logger(__name__)

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return {
        (1, 0): 8,  # Tesla
        (1, 1): 8,
        (1, 2): 8,
        (1, 3): 8,
        (2, 0): 32,  # Fermi
        (2, 1): 48,
        (3, 0): 192,  # Kepler
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,  # Maxwell
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,  # Pascal
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,  # Volta
        (7, 2): 64,
        (7, 5): 64,  # Turing
    }.get((major, minor), 0)


def get_cuda_info():
    libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        logger.warning("could not load any of: " + " ".join(libnames))
        return []

    nGpus = ctypes.c_int()
    name = b" " * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)

    return_obj = []

    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        logger.warning(
            "cuInit failed with error code %d: %s" % (result, error_str.value.decode())
        )
        return []
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        logger.warning(
            "cuDeviceGetCount failed with error code %d: %s"
            % (result, error_str.value.decode())
        )
        return []
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            logger.warning(
                "cuDeviceGet failed with error code %d: %s"
                % (result, error_str.value.decode())
            )
            return []
        print("Device: %d" % i)
        obj = {}
        if (
            cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
            == CUDA_SUCCESS
        ):
            obj["name"] = name.split(b"\0", 1)[0].decode()
        if (
            cuda.cuDeviceComputeCapability(
                ctypes.byref(cc_major), ctypes.byref(cc_minor), device
            )
            == CUDA_SUCCESS
        ):
            obj["capability"] = (int(cc_major.value), int(cc_minor.value))
        if (
            cuda.cuDeviceGetAttribute(
                ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
            )
            == CUDA_SUCCESS
        ):
            obj["cores"] = cores.value
            obj["cuda_core"] = (
                cores.value * ConvertSMVer2Cores(cc_major.value, cc_minor.value)
                or "unknown"
            )
            if (
                cuda.cuDeviceGetAttribute(
                    ctypes.byref(threads_per_core),
                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    device,
                )
                == CUDA_SUCCESS
            ):
                obj["threads"] = cores.value * threads_per_core.value
        if (
            cuda.cuDeviceGetAttribute(
                ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device
            )
            == CUDA_SUCCESS
        ):
            obj["clock"] = clockrate.value / 1000.0
        if (
            cuda.cuDeviceGetAttribute(
                ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
            )
            == CUDA_SUCCESS
        ):
            obj["memory_clock"] = clockrate.value / 1000.0
        result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            # logger.warning("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
        else:
            result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                obj["total_memory"] = totalMem.value / 1024**2
                obj["free_memory"] = freeMem.value / 1024**2
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                logger.warning(
                    "cuMemGetInfo failed with error code %d: %s"
                    % (result, error_str.value.decode())
                )
            cuda.cuCtxDetach(context)

        return_obj.append(obj)

    return return_obj


def get_num_gpus() -> int:
    """
    Get number of gpus using nvidia-smi command
    """
    if "NUM_GPUS" in os.environ:
        num_gpus = os.environ["NUM_GPUS"]
        try:
            num_gpus = int(num_gpus)
            return num_gpus
        except ValueError:
            pass

    try:
        out = subprocess.check_output(["nvidia-smi", "-L"])
        return len(out.decode().splitlines())
    except FileNotFoundError:
        pass

    return -1


def get_gpu_memory(max_tries=50):
    tries = 0
    while tries < max_tries:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            # Parse the output to get memory usage as a list of integers (in MB)
            memory_usage = [int(x) for x in result.stdout.strip().split("\n")]
            return memory_usage
        except Exception as e:
            logger.warning(f"Failed to get GPU memory usage: {e}. Retrying...")
            time.sleep(1)
            tries += 1

    raise RuntimeError(f"Failed to get GPU memory usage after {max_tries} tries.")


def wait_for_memory_release(target_gpu_index, threshold_mb=1024.0, max_tries=100):
    # Wait until memory usage for the specified GPU is below the threshold
    tries = 0
    while tries < max_tries:
        memory_usage = get_gpu_memory()
        if memory_usage[target_gpu_index] < threshold_mb:
            logger.info(
                f"GPU {target_gpu_index} has less than {threshold_mb} MB used. Continuing..."
            )
            return
        else:
            logger.info(
                f"GPU {target_gpu_index} memory used: {memory_usage[target_gpu_index]} MB. Waiting..."
            )
            time.sleep(2)
            tries += 1

    raise RuntimeError(
        f"GPU {target_gpu_index} memory usage is still above {threshold_mb} MB after {max_tries} tries."
    )
