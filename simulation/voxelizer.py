"""
Mesh Voxelizer

Converts triangle mesh to 3D voxel grid for CT simulation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


@dataclass  
class VoxelGrid:
    """
    Represents a 3D voxel grid.
    
    Attributes:
        data: 3D numpy array where non-zero values indicate occupied voxels
        voxel_size: Size of each voxel edge in mm
        origin: (3,) array, world coordinates of the grid origin (corner)
        shape: Grid dimensions (nx, ny, nz)
    """
    data: np.ndarray  # 3D array, dtype varies by use case
    voxel_size: float  # mm per voxel
    origin: np.ndarray  # (3,) world position of grid[0,0,0]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def physical_size(self) -> np.ndarray:
        """Physical dimensions in mm."""
        return np.array(self.shape) * self.voxel_size
    
    def world_to_voxel(self, point: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel indices."""
        return ((point - self.origin) / self.voxel_size).astype(int)
    
    def voxel_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates (voxel center)."""
        return self.origin + (indices + 0.5) * self.voxel_size


class Voxelizer:
    """
    Converts triangle meshes to 3D voxel grids.
    
    Uses trimesh's voxelization functionality which employs ray casting
    to determine interior/exterior voxels.
    """
    
    # Maximum memory allocation for voxel grid (default 2GB)
    MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
    
    def __init__(self, voxel_size: float = 0.5, fill_interior: bool = True,
                 max_memory_gb: float = 2.0):
        """
        Initialize the voxelizer.
        
        Args:
            voxel_size: Edge length of each voxel in mm
            fill_interior: If True, fill interior of watertight meshes
            max_memory_gb: Maximum memory to use in GB
        """
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required for voxelization. "
                "Install it with: pip install trimesh"
            )
        
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
            
        self.voxel_size = voxel_size
        self.fill_interior = fill_interior
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
    
    def get_safe_voxel_size(self, mesh: "trimesh.Trimesh", 
                            desired_voxel_size: float) -> float:
        """
        Calculate a safe voxel size that won't exceed memory limits.
        
        Args:
            mesh: The mesh to voxelize
            desired_voxel_size: Desired voxel size in mm
            
        Returns:
            Safe voxel size (may be larger than desired if memory limited)
        """
        extents = mesh.bounds[1] - mesh.bounds[0]
        
        # Calculate memory for desired voxel size
        shape = np.ceil(extents / desired_voxel_size).astype(int) + 4
        estimated_memory = int(np.prod(shape) * 4)  # float32 = 4 bytes
        
        if estimated_memory <= self.max_memory_bytes:
            return desired_voxel_size
        
        # Calculate minimum voxel size that fits in memory
        # Volume = memory / 4 bytes
        max_voxels = self.max_memory_bytes / 4
        # Cubic root to get approximate dimension
        max_dim = max_voxels ** (1/3)
        
        # Use the largest extent to calculate safe voxel size
        max_extent = np.max(extents)
        safe_voxel_size = max_extent / (max_dim - 4)
        
        return max(safe_voxel_size, desired_voxel_size)
    
    def voxelize(
        self, 
        mesh: "trimesh.Trimesh",
        padding: int = 2,
        auto_adjust: bool = True
    ) -> VoxelGrid:
        """
        Convert a mesh to a voxel grid.
        
        Args:
            mesh: trimesh.Trimesh object to voxelize
            padding: Number of empty voxels to add around the mesh
            auto_adjust: If True, automatically adjust voxel size to fit memory
            
        Returns:
            VoxelGrid containing the voxelized mesh
        """
        pitch = self.voxel_size
        
        # Check and adjust voxel size if needed
        if auto_adjust:
            safe_pitch = self.get_safe_voxel_size(mesh, pitch)
            if safe_pitch > pitch:
                logging.warning(
                    f"Voxel size auto-adjusted from {pitch:.4f} to "
                    f"{safe_pitch:.4f} mm to fit memory limits"
                )
                pitch = safe_pitch
                self.voxel_size = pitch
        
        # Voxelize the mesh
        # This creates a VoxelGrid where True indicates surface voxels
        voxel_grid = mesh.voxelized(pitch=pitch)
        
        # Fill interior if requested and mesh is watertight
        if self.fill_interior and mesh.is_watertight:
            voxel_grid = voxel_grid.fill()
        
        # Get the binary matrix
        matrix = voxel_grid.matrix.astype(np.float32)
        
        # Add padding
        if padding > 0:
            padded_shape = tuple(s + 2 * padding for s in matrix.shape)
            padded_matrix = np.zeros(padded_shape, dtype=np.float32)
            padded_matrix[
                padding:padding+matrix.shape[0],
                padding:padding+matrix.shape[1],
                padding:padding+matrix.shape[2]
            ] = matrix
            matrix = padded_matrix
            
            # Adjust origin for padding
            origin = voxel_grid.transform[:3, 3] - padding * pitch
        else:
            origin = voxel_grid.transform[:3, 3].copy()
        
        return VoxelGrid(
            data=matrix,
            voxel_size=pitch,
            origin=origin
        )
    
    def voxelize_with_resolution(
        self,
        mesh: "trimesh.Trimesh",
        resolution: int = 256,
        padding: int = 2
    ) -> VoxelGrid:
        """
        Voxelize mesh with a target resolution for the longest dimension.
        
        Args:
            mesh: trimesh.Trimesh object to voxelize
            resolution: Target number of voxels along the longest axis
            padding: Number of empty voxels to add around the mesh
            
        Returns:
            VoxelGrid containing the voxelized mesh
        """
        # Calculate voxel size based on mesh extents
        extents = mesh.bounds[1] - mesh.bounds[0]
        max_extent = np.max(extents)
        
        voxel_size = max_extent / (resolution - 2 * padding)
        self.voxel_size = voxel_size
        
        return self.voxelize(mesh, padding=padding)
    
    @staticmethod
    def get_memory_estimate(
        mesh: "trimesh.Trimesh",
        voxel_size: float,
        dtype: np.dtype = np.float32
    ) -> int:
        """
        Estimate memory usage for voxelization.
        
        Args:
            mesh: The mesh to voxelize
            voxel_size: Voxel edge length in mm
            dtype: Data type for the voxel array
            
        Returns:
            Estimated memory usage in bytes
        """
        extents = mesh.bounds[1] - mesh.bounds[0]
        shape = np.ceil(extents / voxel_size).astype(int) + 4  # +4 for padding
        return int(np.prod(shape) * np.dtype(dtype).itemsize)
