"""
STL File Loader

Provides functionality to load and validate STL (Stereolithography) files
for CT simulation.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import tempfile
import hashlib
import logging
import pickle

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from stl import mesh as stl_mesh
    HAS_NUMPY_STL = True
except ImportError:
    HAS_NUMPY_STL = False


# Module-level cache directory
_CACHE_DIR = Path(tempfile.gettempdir()) / "ct_simulation_cache"
_CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class MeshInfo:
    """Information about a loaded mesh."""
    num_vertices: int
    num_faces: int
    bounds_min: np.ndarray  # (3,) array [x_min, y_min, z_min]
    bounds_max: np.ndarray  # (3,) array [x_max, y_max, z_max]
    dimensions: np.ndarray  # (3,) array [width, height, depth] in mm
    center: np.ndarray  # (3,) array [x, y, z] center point
    volume: Optional[float]  # Volume in mm³ (only for watertight meshes)
    is_watertight: bool
    
    def __str__(self) -> str:
        return (
            f"Mesh Info:\n"
            f"  Vertices: {self.num_vertices:,}\n"
            f"  Faces: {self.num_faces:,}\n"
            f"  Dimensions: {self.dimensions[0]:.2f} x {self.dimensions[1]:.2f} x {self.dimensions[2]:.2f} mm\n"
            f"  Volume: {self.volume:.2f} mm³\n" if self.volume else ""
            f"  Watertight: {self.is_watertight}"
        )


class STLLoader:
    """
    Loader for STL (Stereolithography) files.
    
    Supports both ASCII and binary STL formats. Uses trimesh as the primary
    backend with numpy-stl as fallback.
    
    Attributes:
        mesh: The loaded trimesh.Trimesh object
        info: MeshInfo dataclass with mesh statistics
    """
    
    # Class-level cache for quick access
    _mesh_cache: dict = {}
    
    def __init__(self):
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required for STL loading. "
                "Install it with: pip install trimesh"
            )
        self.mesh: Optional[trimesh.Trimesh] = None
        self.info: Optional[MeshInfo] = None
        self._filepath: Optional[Path] = None
        self._cache_key: Optional[str] = None
    
    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Compute MD5 hash of file for cache key."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def _get_cache_path(cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return _CACHE_DIR / f"{cache_key}.glb"
    
    @staticmethod
    def clear_cache() -> int:
        """
        Clear all cached mesh files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        try:
            for cache_file in _CACHE_DIR.glob("*.glb"):
                cache_file.unlink()
                count += 1
            logging.info(f"Cleared {count} cached mesh files from {_CACHE_DIR}")
        except Exception as e:
            logging.warning(f"Failed to clear cache: {e}")
        return count
    
    def _save_to_cache(self) -> None:
        """Save current mesh to disk cache."""
        if self.mesh is None or self._cache_key is None:
            return
        try:
            cache_path = self._get_cache_path(self._cache_key)
            self.mesh.export(cache_path, file_type='glb')
            logging.info(f"Mesh cached to: {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to cache mesh: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[trimesh.Trimesh]:
        """Load mesh from disk cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                mesh = trimesh.load(cache_path, file_type='glb')
                # Ensure we have a Trimesh, not a Scene
                mesh = self._ensure_trimesh(mesh)
                logging.info(f"Mesh loaded from cache: {cache_path}")
                return mesh
            except Exception as e:
                logging.warning(f"Failed to load from cache: {e}")
        return None
    
    def _ensure_trimesh(self, mesh_or_scene) -> trimesh.Trimesh:
        """
        Ensure we have a Trimesh object, extracting from Scene if necessary.
        
        Args:
            mesh_or_scene: Either a Trimesh or Scene object
            
        Returns:
            trimesh.Trimesh object
            
        Raises:
            ValueError: If no valid mesh can be extracted
        """
        if isinstance(mesh_or_scene, trimesh.Trimesh):
            return mesh_or_scene
            
        if isinstance(mesh_or_scene, trimesh.Scene):
            logging.info("Loaded data is a Scene, extracting mesh geometry...")
            meshes = [g for g in mesh_or_scene.geometry.values() 
                     if isinstance(g, trimesh.Trimesh)]
            
            if not meshes:
                raise ValueError("No valid meshes found in file (empty scene)")
            
            if len(meshes) == 1:
                return meshes[0]
            else:
                return trimesh.util.concatenate(meshes)
        
        raise ValueError(f"Unexpected mesh type: {type(mesh_or_scene)}")
    
    def load(
        self, 
        filepath: str | Path, 
        use_cache: bool = True,
        auto_normalize: bool = True,
        target_size_mm: float = 100.0
    ) -> trimesh.Trimesh:
        """
        Load an STL file.
        
        Args:
            filepath: Path to the STL file
            use_cache: Whether to use disk caching (default True)
            auto_normalize: Whether to auto-scale mesh to target size (default True)
            target_size_mm: Target max dimension in mm when normalizing (default 100)
            
        Returns:
            trimesh.Trimesh object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a valid STL
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"STL file not found: {filepath}")
        
        if filepath.suffix.lower() != '.stl':
            raise ValueError(f"Expected .stl file, got: {filepath.suffix}")
        
        # Compute cache key
        self._cache_key = self._compute_file_hash(filepath)
        logging.info(f"Loading STL: {filepath.name} (hash: {self._cache_key[:8]}...)")
        
        # Try to load from cache first
        if use_cache:
            cached_mesh = self._load_from_cache(self._cache_key)
            if cached_mesh is not None:
                self.mesh = cached_mesh
                # Apply normalization to cached mesh too (in case cache was from old version)
                if auto_normalize:
                    self._normalize_mesh(target_size_mm)
                self._filepath = filepath
                self._compute_info()
                logging.info("Using cached mesh data.")
                return self.mesh
        
        # Load from STL file
        logging.info("Loading from STL file (not cached)...")
        try:
            self.mesh = trimesh.load(filepath, file_type='stl')
        except Exception as e:
            raise ValueError(f"Failed to load STL file: {e}")
        
        # Ensure we have a Trimesh, not a Scene
        self.mesh = self._ensure_trimesh(self.mesh)
        
        # Final verification
        if isinstance(self.mesh, trimesh.Scene):
            raise ValueError("Failed to convert Scene to Mesh")
        
        # Auto-normalize mesh size if dimensions are unreasonable
        if auto_normalize:
            self._normalize_mesh(target_size_mm)
            
        self._filepath = filepath
        self._compute_info()
        
        # Save to cache for future use
        if use_cache:
            self._save_to_cache()
        
        return self.mesh
    
    def _normalize_mesh(self, target_size_mm: float = 100.0) -> None:
        """
        Normalize mesh to exactly fit target size.
        
        Always scales the mesh so that its largest dimension equals
        target_size_mm, and centers it at the origin.
        
        Args:
            target_size_mm: Target max dimension in mm
        """
        if self.mesh is None:
            return
        
        bounds = self.mesh.bounds
        current_size = np.max(bounds[1] - bounds[0])
        
        if current_size <= 0:
            logging.warning("Mesh has zero size, skipping normalization")
            return
        
        # Always scale to target size
        scale_factor = target_size_mm / current_size
        
        # Apply scaling
        self.mesh.apply_scale(scale_factor)
        
        # Center at origin
        centroid = self.mesh.centroid
        self.mesh.apply_translation(-centroid)
        
        new_size = np.max(self.mesh.bounds[1] - self.mesh.bounds[0])
        logging.info(f"Mesh normalized: {current_size:.2f} -> {new_size:.1f} mm (scale: {scale_factor:.4f})")
    
    def reload_from_cache(self) -> Optional[trimesh.Trimesh]:
        """
        Reload mesh from cache without reading original file.
        Useful for re-voxelization with different parameters.
        
        Returns:
            trimesh.Trimesh if cache exists, None otherwise
        """
        if self._cache_key is None:
            logging.warning("No cache key available. Load a file first.")
            return None
        
        cached_mesh = self._load_from_cache(self._cache_key)
        if cached_mesh is not None:
            self.mesh = cached_mesh
            self._compute_info()
            logging.info("Mesh reloaded from cache successfully.")
            return self.mesh
        
        logging.warning("Cache miss - mesh not found in cache.")
        return None
    
    def _compute_info(self) -> None:
        """Compute mesh statistics and store in self.info."""
        if self.mesh is None:
            return
            
        bounds = self.mesh.bounds  # (2, 3) array: [min, max]
        bounds_min = bounds[0]
        bounds_max = bounds[1]
        dimensions = bounds_max - bounds_min
        center = (bounds_min + bounds_max) / 2
        
        # Volume calculation (only valid for watertight meshes)
        is_watertight = self.mesh.is_watertight
        volume = self.mesh.volume if is_watertight else None
        
        self.info = MeshInfo(
            num_vertices=len(self.mesh.vertices),
            num_faces=len(self.mesh.faces),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            dimensions=dimensions,
            center=center,
            volume=volume,
            is_watertight=is_watertight,
        )
    
    def get_vertices(self) -> np.ndarray:
        """Get mesh vertices as (N, 3) array."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")
        return self.mesh.vertices.copy()
    
    def get_faces(self) -> np.ndarray:
        """Get mesh faces as (M, 3) array of vertex indices."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")
        return self.mesh.faces.copy()
    
    def get_face_normals(self) -> np.ndarray:
        """Get face normals as (M, 3) array."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")
        return self.mesh.face_normals.copy()
    
    def center_mesh(self) -> None:
        """Translate mesh so its center is at the origin."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")
        self.mesh.vertices -= self.info.center
        self._compute_info()
    
    def scale_mesh(self, factor: float) -> None:
        """Scale mesh by a factor."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")
        self.mesh.vertices *= factor
        self._compute_info()
    
    @property
    def filepath(self) -> Optional[Path]:
        """Get the path of the loaded file."""
        return self._filepath
