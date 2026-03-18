#!/usr/bin/env python3
"""
VK2Torch CUDA External Memory Interoperability Layer
Provides zero-copy integration between Vulkan external memory and CUDA/PyTorch tensors.

This module contains CUDA external memory functions extracted from the working
vk2torch client implementations, specifically for:
1. External memory import via file descriptors
2. Timeline semaphore synchronization
3. Depth buffer format conversion (D24 to float)
4. Zero-copy tensor creation using DLPack
"""

import ctypes
import ctypes.util
import os
import sys
import numpy as np
from typing import Optional, Tuple, Any

# CUDA constants and error codes
CUDA_SUCCESS = 0
CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9

class CudaError(Exception):
    """CUDA operation failed"""
    pass

class CudaContext:
    """CUDA context manager for external memory operations"""
    
    def __init__(self):
        self.libcuda = None
        self.context = None
        self.device = None
        self._initialize()
    
    def _initialize(self):
        """Initialize CUDA library and context"""
        try:
            # Load CUDA driver library
            self.libcuda = ctypes.CDLL(ctypes.util.find_library('cuda'))
            if not self.libcuda:
                raise CudaError("CUDA driver library not found")
            
            # Initialize CUDA
            result = self.libcuda.cuInit(0)
            if result != CUDA_SUCCESS:
                raise CudaError(f"cuInit failed with error {result}")
            
            # Create context
            self.context = ctypes.c_void_p()
            self.device = ctypes.c_int(0)
            result = self.libcuda.cuCtxCreate_v2(
                ctypes.byref(self.context), 0, self.device
            )
            if result != CUDA_SUCCESS:
                raise CudaError(f"cuCtxCreate failed with error {result}")
            
        except Exception as e:
            raise CudaError(f"CUDA initialization failed: {e}")
    
    def __del__(self):
        """Cleanup CUDA context"""
        if self.libcuda and self.context:
            try:
                self.libcuda.cuCtxDestroy_v2(self.context)
            except:
                pass

# Global CUDA context (lazy initialization)
_cuda_context: Optional[CudaContext] = None

def get_cuda_context() -> CudaContext:
    """Get or create the global CUDA context"""
    global _cuda_context
    if _cuda_context is None:
        _cuda_context = CudaContext()
    return _cuda_context




CUdeviceptr = ctypes.c_uint64

class _Win32Pair(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_void_p),
        ("name",   ctypes.c_void_p),
    ]

class _HandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _Win32Pair),
        ("nvSciBufObject", ctypes.c_void_p),
    ]

class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),                 # CUexternalMemoryHandleType
        ("handle", _HandleUnion),
        ("size", ctypes.c_uint64),               # allocation size (NOT buffer size)
        ("flags", ctypes.c_uint),                # 0 or CUDA_EXTERNAL_MEMORY_DEDICATED
        ("reserved", ctypes.c_uint * 16) ,
    ]





class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(ctypes.Structure):
    """CUDA external memory buffer descriptor"""
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16)
    ]

class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC(ctypes.Structure):
    class Handle(ctypes.Union):
        _fields_ = [
            ("fd", ctypes.c_int),
            ("win32", ctypes.c_void_p),  # Not used on Linux
            ("nvSciSyncObj", ctypes.c_void_p),  # Not used
        ]
    
    _fields_ = [
        ("type", ctypes.c_uint),
        ("handle", Handle),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]
        


class _WaitFence(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint64),
        ("reserved", ctypes.c_uint * 16),   # ★ 必须有
    ]

class _WaitKeyedMutex(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint),
        ("timeoutMs", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 14),
    ]

class _WaitParamsUnion(ctypes.Union):
    _fields_ = [("fence", _WaitFence), ("keyedMutex", _WaitKeyedMutex)]

class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS(ctypes.Structure):
    _fields_ = [
        ("params", _WaitParamsUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _SigFence(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint64),
        ("reserved", ctypes.c_uint * 16),   # ★ 必须有
    ]

class _SigKeyedMutex(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint),
        ("timeoutMs", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 14),   # 保持整体大小一致
    ]

class _SigParamsUnion(ctypes.Union):
    _fields_ = [("fence", _SigFence), ("keyedMutex", _SigKeyedMutex)]

class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS(ctypes.Structure):
    _fields_ = [
        ("params", _SigParamsUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]

def import_ext_memory_fd(fd: int, size: int, is_dedicated: bool = False) -> Tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """
    Import Vulkan external memory via file descriptor into CUDA
    
    Args:
        fd: File descriptor for external memory
        size: Size of memory in bytes
        is_dedicated: Whether the memory is dedicated allocation
    
    Returns:
        Tuple of (external memory handle, device pointer)
    
    Raises:
        CudaError: If import fails
    """
    ctx = get_cuda_context()
    
    # Set up external memory descriptor
    desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
    ctypes.memset(ctypes.byref(desc), 0, ctypes.sizeof(desc))
    desc.type = CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
    desc.handle.fd = fd
    desc.size = size
    desc.flags = 0x01 if is_dedicated else 0x00
    
    # Import external memory
    ext_mem = ctypes.c_void_p()
    ctx.libcuda.cuImportExternalMemory.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC)
    ]
    ctx.libcuda.cuImportExternalMemory.restype = ctypes.c_int
    
    result = ctx.libcuda.cuImportExternalMemory(
        ctypes.byref(ext_mem), ctypes.byref(desc)
    )
    
    if result != CUDA_SUCCESS:
        raise CudaError(f"cuImportExternalMemory failed with error {result}")
    
    # Get device pointer for the full buffer
    dev_ptr = get_mapped_buffer_pointer(ext_mem, 0, size)
    
    return ext_mem, dev_ptr

def import_timeline_semaphore_fd(fd: int) -> ctypes.c_void_p:
    """
    Import Vulkan timeline semaphore via file descriptor into CUDA
    
    Args:
        fd: File descriptor for timeline semaphore
    
    Returns:
        CUDA external semaphore handle
    
    Raises:
        CudaError: If import fails
    """
    ctx = get_cuda_context()
    
    # Set up external semaphore descriptor
    desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
    ctypes.memset(ctypes.byref(desc), 0, ctypes.sizeof(desc))
    desc.type = CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
    desc.handle.fd = fd
    desc.flags = 0x00
    
    # Import external semaphore
    ext_sem = ctypes.c_void_p()
    ctx.libcuda.cuImportExternalSemaphore.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)
    ]
    ctx.libcuda.cuImportExternalSemaphore.restype = ctypes.c_int
    
    result = ctx.libcuda.cuImportExternalSemaphore(
        ctypes.byref(ext_sem), ctypes.byref(desc)
    )
    
    if result != CUDA_SUCCESS:
        raise CudaError(f"cuImportExternalSemaphore failed with error {result}")
    
    return ext_sem

def get_mapped_buffer_pointer(ext_mem: ctypes.c_void_p, offset: int, size: int) -> ctypes.c_void_p:
    """
    Get device pointer for mapped external memory buffer
    
    Args:
        ext_mem: External memory handle from import_ext_memory_fd
        offset: Offset into the memory
        size: Size of the buffer
    
    Returns:
        CUDA device pointer
    
    Raises:
        CudaError: If mapping fails
    """
    ctx = get_cuda_context()
    
    # Set up buffer descriptor
    buf_desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
    buf_desc.offset = offset
    buf_desc.size = size
    buf_desc.flags = 0
    
    # Map buffer
    dev_ptr = ctypes.c_void_p()
    ctx.libcuda.cuExternalMemoryGetMappedBuffer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC)
    ]
    ctx.libcuda.cuExternalMemoryGetMappedBuffer.restype = ctypes.c_int
    
    result = ctx.libcuda.cuExternalMemoryGetMappedBuffer(
        ctypes.byref(dev_ptr), ext_mem, ctypes.byref(buf_desc)
    )
    
    if result != CUDA_SUCCESS:
        raise CudaError(f"cuExternalMemoryGetMappedBuffer failed with error {result}")
    
    return dev_ptr

def wait_timeline(ext_sem: ctypes.c_void_p, value: int, stream_ptr: int) -> None:
    """
    Wait for timeline semaphore to reach specified value
    
    Args:
        ext_sem: External semaphore handle from import_timeline_semaphore_fd
        value: Timeline value to wait for
        stream_ptr: CUDA stream pointer (use 0 for null stream)
    
    Raises:
        CudaError: If wait fails
    """
    ctx = get_cuda_context()
    
    # Set up wait parameters
    wait_params = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
    ctypes.memset(ctypes.byref(wait_params), 0, ctypes.sizeof(wait_params))
    wait_params.params.fence.value = ctypes.c_uint64(value)
    wait_params.flags = 0
    
    # Wait on semaphore
    ctx.libcuda.cuWaitExternalSemaphoresAsync.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS),
        ctypes.c_uint, ctypes.c_void_p
    ]
    ctx.libcuda.cuWaitExternalSemaphoresAsync.restype = ctypes.c_int
    
    result = ctx.libcuda.cuWaitExternalSemaphoresAsync(
        ctypes.byref(ext_sem), ctypes.byref(wait_params), 1,
        ctypes.c_void_p(0)# ctypes.c_void_p(stream_ptr)
    )
    
    if result != CUDA_SUCCESS:
        raise CudaError(f"cuWaitExternalSemaphoresAsync failed with error {result}")

def signal_timeline(ext_sem: ctypes.c_void_p, value: int, stream_ptr: int) -> None:
    """
    Signal timeline semaphore with specified value
    
    Args:
        ext_sem: External semaphore handle from import_timeline_semaphore_fd
        value: Timeline value to signal
        stream_ptr: CUDA stream pointer (use 0 for null stream)
    
    Raises:
        CudaError: If signal fails
    """
    ctx = get_cuda_context()
    
    # Set up signal parameters
    signal_params = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
    ctypes.memset(ctypes.byref(signal_params), 0, ctypes.sizeof(signal_params))
    signal_params.params.fence.value = value
    signal_params.flags = 0
    
    #   self.cuda.cuSignalExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), ctypes.c_uint, ctypes.c_void_p]
    #     self.cuda.cuSignalExternalSemaphoresAsync.restype = ctypes.c_int

    # Signal semaphore
    ctx.libcuda.cuSignalExternalSemaphoresAsync.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS),
        ctypes.c_uint, ctypes.c_void_p
    ]
    ctx.libcuda.cuSignalExternalSemaphoresAsync.restype = ctypes.c_int
    
    result = ctx.libcuda.cuSignalExternalSemaphoresAsync(
        ctypes.byref(ext_sem), ctypes.byref(signal_params), 1,
        ctypes.c_void_p(stream_ptr)
    )
    
    if result != CUDA_SUCCESS:
        raise CudaError(f"cuSignalExternalSemaphoresAsync failed with error {result}")

def make_pitched_cupy_array(device_ptr: int, row_pitch_bytes: int, width: int, height: int, dtype: np.dtype) -> Any:
    """
    Create CuPy array from CUDA device pointer with row pitch
    
    Args:
        device_ptr: CUDA device pointer as integer address
        row_pitch_bytes: Row pitch in bytes
        width: Width in elements
        height: Height in rows
        dtype: NumPy dtype for the array
    
    Returns:
        CuPy array with proper strides
    
    Raises:
        ImportError: If CuPy not available
        ValueError: If parameters invalid
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy is required for pitched array creation")
    
    # if row_pitch_bytes < width * dtype.itemsize:
    #     raise ValueError(f"Row pitch {row_pitch_bytes} too small for width {width} * itemsize {dtype.itemsize}")
    
    dt = np.dtype(dtype)
    itemsize = int(dt.itemsize)

    # Calculate strides: row stride in bytes, column stride is itemsize
    strides = (row_pitch_bytes, itemsize)
    
    # Create memory pointer from integer device pointer
    memptr = cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(device_ptr, height * row_pitch_bytes, owner=None), 0
    )
    
    # Create array with custom strides
    array = cp.ndarray(
        shape=(height, width),
        dtype=dtype,
        memptr=memptr,
        strides=strides
    )

    umem = cp.cuda.UnownedMemory(
        device_ptr,
        height * row_pitch_bytes,  # Total size 
        owner=None
    )
    mptr = cp.cuda.MemoryPointer(umem, 0)
    
    # Create raw uint32 array with proper strides for row pitch
    raw32 = cp.ndarray(
        (height, width), 
        dtype=cp.uint32, 
        memptr=mptr,
        strides=strides  # (row_pitch_bytes, bytes_per_pixel)
    )


    return raw32

def depth_d24_to_float(d24_array: Any) -> Any:
    """
    Convert D24 depth format to float32
    
    D24 format packs 24-bit depth values into 32-bit words.
    The depth value is in the upper 24 bits.
    
    Args:
        d24_array: CuPy array with uint32 D24 data
    
    Returns:
        CuPy array with float32 depth values [0.0, 1.0]
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy is required for depth conversion")
    
    if not hasattr(d24_array, 'dtype') or d24_array.dtype != cp.uint32:
        raise ValueError("Input must be CuPy uint32 array")
    
    # Extract 24-bit depth from upper 24 bits and normalize
    # Right shift by 8 to move depth to lower bits, then normalize by 2^24-1
    depth_24bit = (d24_array & 0x00FFFFFF)
    depth_float = depth_24bit.astype(cp.float32) / float(0xFFFFFF)
    
    return depth_float

def to_torch(cupy_array: Any) -> Any:
    """
    Convert CuPy array to PyTorch tensor using zero-copy DLPack
    
    Args:
        cupy_array: CuPy array to convert
    
    Returns:
        PyTorch tensor sharing the same memory
    
    Raises:
        ImportError: If PyTorch not available
    """
    try:
        import torch
        import cupy as cp
    except ImportError:
        raise ImportError("PyTorch and CuPy are required for tensor conversion")
    
    if not hasattr(cupy_array, '__dlpack__'):
        raise ValueError("Input must support DLPack protocol")
    
    # Zero-copy conversion via DLPack
    torch_tensor = torch.from_dlpack(cupy_array.__dlpack__())
    
    return torch_tensor

def create_zero_copy_depth_tensor(fd: int, width: int, height: int, pitch: int, 
                                 is_dedicated: bool = False) -> Tuple[Any, ctypes.c_void_p]:
    """
    Create zero-copy PyTorch tensor from Vulkan depth buffer file descriptor
    
    Args:
        fd: File descriptor for depth buffer external memory
        width: Width in pixels
        height: Height in pixels
        pitch: Row pitch in bytes
        is_dedicated: Whether memory is dedicated allocation
    
    Returns:
        Tuple of (PyTorch tensor with float32 depth, external memory handle)
    
    Raises:
        CudaError: If CUDA operations fail
        ImportError: If required libraries not available
    """
    # Calculate buffer size
    buffer_size = height * pitch
    
    # Import external memory and get device pointer
    ext_mem, dev_ptr = import_ext_memory_fd(fd, buffer_size, is_dedicated)
    
    # Create pitched CuPy array for D24 format (uint32) using new signature
    d24_array = make_pitched_cupy_array(dev_ptr.value, pitch, width, height, np.uint32)
    
    # Convert D24 to float32
    depth_float = depth_d24_to_float(d24_array)
    
    # Convert to PyTorch tensor
    torch_tensor = to_torch(depth_float)
    
    return torch_tensor, ext_mem

# Convenience functions for testing
def test_cuda_availability() -> bool:
    """Test if CUDA is available and working"""
    try:
        ctx = get_cuda_context()
        return True
    except CudaError:
        return False

def get_cuda_driver_version() -> Optional[str]:
    """Get CUDA driver version string"""
    try:
        ctx = get_cuda_context()
        version = ctypes.c_int()
        result = ctx.libcuda.cuDriverGetVersion(ctypes.byref(version))
        if result == CUDA_SUCCESS:
            major = version.value // 1000
            minor = (version.value % 1000) // 10
            return f"{major}.{minor}"
        return None
    except:
        return None

if __name__ == "__main__":
    # Simple test
    print("VK2Torch CUDA Module Test")
    print("=" * 40)
    
    if test_cuda_availability():
        print("✅ CUDA is available")
        version = get_cuda_driver_version()
        if version:
            print(f"✅ CUDA driver version: {version}")
        else:
            print("⚠️  Could not get CUDA driver version")
    else:
        print("❌ CUDA is not available")
    
    # Test imports
    try:
        import cupy as cp
        print(f"✅ CuPy version: {cp.__version__}")
    except ImportError:
        print("❌ CuPy not available")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not available")