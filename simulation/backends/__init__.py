"""
Simulation Backends

Provides CPU and GPU backends for Radon/IRadon transforms.
"""

from .base import SimulationBackend
from .cpu_backend import CPUBackend
from .gpu_backend import GPUBackend, HAS_CUPY


def get_backend(use_gpu: bool = False) -> SimulationBackend:
    """
    Get the appropriate simulation backend.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        SimulationBackend instance (GPU if available and requested, else CPU)
    """
    if use_gpu and HAS_CUPY:
        return GPUBackend()
    return CPUBackend()


__all__ = [
    'SimulationBackend',
    'CPUBackend', 
    'GPUBackend',
    'HAS_CUPY',
    'get_backend',
]
