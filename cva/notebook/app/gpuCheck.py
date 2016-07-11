#!/bin/bash

import ctypes
import platform
import sys

"""
cudart.py: used to access pars of the CUDA runtime library.
Most of this code was lifted from the pystream project (it's BSD licensed):
http://code.google.com/p/pystream

Note that this is likely to only work with CUDA 2.3
To extend to other versions, you may need to edit the DeviceProp Class
"""

cudaSuccess = 0
errorDict = {
    1: 'MissingConfigurationError',
    2: 'MemoryAllocationError',
    3: 'InitializationError',
    4: 'LaunchFailureError',
    5: 'PriorLaunchFailureError',
    6: 'LaunchTimeoutError',
    7: 'LaunchOutOfResourcesError',
    8: 'InvalidDeviceFunctionError',
    9: 'InvalidConfigurationError',
    10: 'InvalidDeviceError',
    11: 'InvalidValueError',
    12: 'InvalidPitchValueError',
    13: 'InvalidSymbolError',
    14: 'MapBufferObjectFailedError',
    15: 'UnmapBufferObjectFailedError',
    16: 'InvalidHostPointerError',
    17: 'InvalidDevicePointerError',
    18: 'InvalidTextureError',
    19: 'InvalidTextureBindingError',
    20: 'InvalidChannelDescriptorError',
    21: 'InvalidMemcpyDirectionError',
    22: 'AddressOfConstantError',
    23: 'TextureFetchFailedError',
    24: 'TextureNotBoundError',
    25: 'SynchronizationError',
    26: 'InvalidFilterSettingError',
    27: 'InvalidNormSettingError',
    28: 'MixedDeviceExecutionError',
    29: 'CudartUnloadingError',
    30: 'UnknownError',
    31: 'NotYetImplementedError',
    32: 'MemoryValueTooLargeError',
    33: 'InvalidResourceHandleError',
    34: 'NotReadyError',
    0x7f: 'StartupFailureError',
    10000: 'ApiFailureBaseError'}

try:
    if platform.system() == "Microsoft":
        _libcudart = ctypes.windll.LoadLibrary('cudart.dll')
    elif platform.system() == "Darwin":
        _libcudart = ctypes.cdll.LoadLibrary('/usr/local/cuda/lib/libcudart.dylib')
    else:
        _libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
    _libcudart_error = None
except OSError, e:
    _libcudart_error = e
    _libcudart = None


def _checkCudaStatus(status):
    if status != cudaSuccess:
        eClassString = errorDict[status]
        # Get the class by name from the top level of this module
        eClass = globals()[eClassString]
        raise eClass()


def _checkDeviceNumber(device):
    assert isinstance(device, int), "device number must be an int"
    assert device >= 0, "device number must be greater than 0"
    assert device < 2 ** 8 - 1, "device number must be < 255"


# cudaDeviceProp
class DeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", 256 * ctypes.c_char),  # < ASCII string identifying device
        ("totalGlobalMem", ctypes.c_size_t),  # < Global memory available on device in bytes
        ("sharedMemPerBlock", ctypes.c_size_t),  # < Shared memory available per block in bytes
        ("regsPerBlock", ctypes.c_int),  # < 32-bit registers available per block
        ("warpSize", ctypes.c_int),  # < Warp size in threads
        ("memPitch", ctypes.c_size_t),  # < Maximum pitch in bytes allowed by memory copies
        ("maxThreadsPerBlock", ctypes.c_int),  # < Maximum number of threads per block
        ("maxThreadsDim", 3 * ctypes.c_int),  # < Maximum size of each dimension of a block
        ("maxGridSize", 3 * ctypes.c_int),  # < Maximum size of each dimension of a grid
        ("clockRate", ctypes.c_int),  # < Clock frequency in kilohertz
        ("totalConstMem", ctypes.c_size_t),  # < Constant memory available on device in bytes
        ("major", ctypes.c_int),  # < Major compute capability
        ("minor", ctypes.c_int),  # < Minor compute capability
        ("textureAlignment", ctypes.c_size_t),  # < Alignment requirement for textures
        ("deviceOverlap", ctypes.c_int),  # < Device can concurrently copy memory and execute a kernel
        ("multiProcessorCount", ctypes.c_int),  # < Number of multiprocessors on device
        ("kernelExecTimeoutEnabled", ctypes.c_int),  # < Specified whether there is a run time limit on kernels
        ("integrated", ctypes.c_int),  # < Device is integrated as opposed to discrete
        ("canMapHostMemory", ctypes.c_int),  # < Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        ("computeMode", ctypes.c_int),  # < Compute mode (See ::cudaComputeMode)
        ("__cudaReserved", 36 * ctypes.c_int),
    ]

    def __str__(self):
        return """NVidia GPU Specifications:
    Name: %s
    Total global mem: %i
    Shared mem per block: %i
    Registers per block: %i
    Warp size: %i
    Mem pitch: %i
    Max threads per block: %i
    Max treads dim: (%i, %i, %i)
    Max grid size: (%i, %i, %i)
    Total const mem: %i
    Compute capability: %i.%i
    Clock Rate (GHz): %f
    Texture alignment: %i
""" % (self.name, self.totalGlobalMem, self.sharedMemPerBlock,
       self.regsPerBlock, self.warpSize, self.memPitch,
       self.maxThreadsPerBlock,
       self.maxThreadsDim[0], self.maxThreadsDim[1], self.maxThreadsDim[2],
       self.maxGridSize[0], self.maxGridSize[1], self.maxGridSize[2],
       self.totalConstMem, self.major, self.minor,
       float(self.clockRate) / 1.0e6, self.textureAlignment)


def cudaGetDeviceCount():
    if _libcudart is None: return 0
    deviceCount = ctypes.c_int()
    status = _libcudart.cudaGetDeviceCount(ctypes.byref(deviceCount))
    _checkCudaStatus(status)
    return deviceCount.value


def getDeviceProperties(device):
    if _libcudart is None: return None
    _checkDeviceNumber(device)
    props = DeviceProp()
    status = _libcudart.cudaGetDeviceProperties(ctypes.byref(props), device)
    _checkCudaStatus(status)
    return props


def getDriverVersion():
    if _libcudart is None: return None
    version = ctypes.c_int()
    _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    v = "%d.%d" % (version.value // 1000,
                   version.value % 100)
    return v


def getRuntimeVersion():
    if _libcudart is None: return None
    version = ctypes.c_int()
    _libcudart.cudaRuntimeGetVersion(ctypes.byref(version))
    v = "%d.%d" % (version.value // 1000,
                   version.value % 100)
    return v


def getGpuCount():
    count = 0
    for ii in range(cudaGetDeviceCount()):
        props = getDeviceProperties(ii)
        if props.major != 9999: count += 1
    return count


def getLoadError():
    return _libcudart_error


version = getDriverVersion()
if version is not None and not version.startswith('2.3'):
    sys.stdout.write("WARNING: Driver version %s may not work with %s\n" %
                     (version, sys.argv[0]))

version = getRuntimeVersion()
if version is not None and not version.startswith('2.3'):
    sys.stdout.write("WARNING: Runtime version %s may not work with %s\n" %
                     (version, sys.argv[0]))
