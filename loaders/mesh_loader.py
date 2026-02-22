"""
Generic Mesh Loader (STL, PLY, OBJ, etc.)

Provides unified functionality to load and validate 3D mesh files
for CT simulation. Supports multiple formats via trimesh backend.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Tuple, Optional
import numpy as np
import tempfile
import hashlib
import logging
import os
import time
import uuid

from core import BaseLoader, ScientificData

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# Supported mesh file extensions
SUPPORTED_EXTENSIONS = {'.stl', '.ply', '.obj', '.off', '.glb', '.gltf'}

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
    file_format: str  # Original file format (e.g., 'stl', 'ply', 'obj')
    
    def __str__(self) -> str:
        vol_str = f"  Volume: {self.volume:.2f} mm³\n" if self.volume else ""
        return (
            f"Mesh Info ({self.file_format.upper()}):\n"
            f"  Vertices: {self.num_vertices:,}\n"
            f"  Faces: {self.num_faces:,}\n"
            f"  Dimensions: {self.dimensions[0]:.2f} x {self.dimensions[1]:.2f} x {self.dimensions[2]:.2f} mm\n"
            f"{vol_str}"
            f"  Watertight: {self.is_watertight}"
        )


class MeshLoader(BaseLoader[Any, dict[str, Any]]):
    """
    Unified loader for 3D mesh files (STL, PLY, OBJ, OFF, GLB, GLTF).
    
    Uses trimesh as the backend for loading all supported formats.
    Provides normalization, caching, and mesh information extraction.
    
    Attributes:
        mesh: The loaded trimesh.Trimesh object
        info: MeshInfo dataclass with mesh statistics
    
    Supported Formats:
        - STL: Stereolithography (ASCII and binary)
        - PLY: Polygon File Format (ASCII and binary)
        - OBJ: Wavefront OBJ
        - OFF: Object File Format
        - GLB/GLTF: GL Transmission Format
    """
    
    # Class-level cache for quick access
    _mesh_cache: dict = {}
    
    def __init__(self):
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required for mesh loading. "
                "Install it with: pip install trimesh"
            )
        self.mesh: Optional[trimesh.Trimesh] = None
        self.info: Optional[MeshInfo] = None
        self._filepath: Optional[Path] = None
        self._cache_key: Optional[str] = None
        self._file_format: str = ""
    
    @staticmethod
    def get_supported_extensions() -> set:
        """Get set of supported file extensions (with leading dot)."""
        return SUPPORTED_EXTENSIONS.copy()
    
    @staticmethod
    def get_file_filter() -> str:
        """Get file filter string for file dialogs."""
        extensions = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        return f"3D Mesh Files ({extensions})"
    
    @staticmethod
    def is_supported(filepath: str | Path) -> bool:
        """Check if a file extension is supported."""
        return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS

    def can_load(self, source: str) -> bool:
        """Check whether the path extension is supported."""
        return self.is_supported(source)
    
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
    def _get_temp_cache_path(cache_key: str) -> Path:
        """
        Get a process-safe temporary cache path in the same directory.

        Writing to a sibling temp file and then replacing the target path
        allows an atomic cache transaction across concurrent processes.
        """
        token = uuid.uuid4().hex
        return _CACHE_DIR / f"{cache_key}.{token}.tmp.glb"
    
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
            # Best-effort cleanup for interrupted temp transactions.
            for temp_file in _CACHE_DIR.glob("*.tmp.glb"):
                temp_file.unlink()
                count += 1
            logging.info(f"Cleared {count} cached mesh files from {_CACHE_DIR}")
        except OSError as exc:
            logging.warning("Failed to clear cache: %s", exc)
        return count
    
    def _save_to_cache(self) -> None:
        """Save current mesh to disk cache."""
        if self.mesh is None or self._cache_key is None:
            return
        temp_path: Optional[Path] = None
        try:
            cache_path = self._get_cache_path(self._cache_key)
            temp_path = self._get_temp_cache_path(self._cache_key)

            # Transaction step 1: write full payload to a unique temp file.
            self.mesh.export(temp_path, file_type='glb')

            # Transaction step 2: atomically publish completed file.
            # On Windows, replace may transiently fail if another process/thread
            # is touching the same cache path; retry briefly to smooth races.
            last_exc: Optional[OSError] = None
            for attempt in range(1, 6):
                try:
                    os.replace(temp_path, cache_path)
                    last_exc = None
                    break
                except PermissionError as exc:
                    last_exc = exc
                except OSError as exc:
                    # WinError 5/32/33 => access denied/sharing violation.
                    if getattr(exc, "winerror", None) in (5, 32, 33):
                        last_exc = exc
                    else:
                        raise

                if attempt < 5:
                    time.sleep(0.02 * attempt)

            if last_exc is not None:
                raise last_exc
            logging.info(f"Mesh cached to: {cache_path}")
        except (OSError, ValueError, RuntimeError) as exc:
            logging.warning("Failed to cache mesh: %s", exc)
        finally:
            # If transaction failed before replace(), remove orphan temp file.
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    
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
            except (OSError, ValueError, RuntimeError, EOFError) as exc:
                logging.warning("Failed to load from cache: %s", exc)
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
                logging.info(f"Combining {len(meshes)} meshes into single mesh")
                return trimesh.util.concatenate(meshes)
        
        raise ValueError(f"Unexpected mesh type: {type(mesh_or_scene)}")
    
    def load(
        self, 
        filepath: str | Path, 
        use_cache: bool = True,
        auto_normalize: bool = True,
        target_size_mm: float = 100.0
    ) -> ScientificData[Any, dict[str, Any]]:
        """
        Load a 3D mesh file.
        
        Args:
            filepath: Path to the mesh file (STL, PLY, OBJ, OFF, GLB, GLTF)
            use_cache: Whether to use disk caching (default True)
            auto_normalize: Whether to auto-scale mesh to target size (default True)
            target_size_mm: Target max dimension in mm when normalizing (default 100)
            
        Returns:
            ScientificData containing mesh and mesh metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported or invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")
        
        extension = filepath.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {supported}"
            )
        
        self._file_format = extension[1:]  # Remove leading dot
        
        # Compute cache key
        self._cache_key = self._compute_file_hash(filepath)
        logging.info(f"Loading mesh: {filepath.name} (format: {self._file_format.upper()}, hash: {self._cache_key[:8]}...)")
        
        # Try to load from cache first
        if use_cache:
            cached_mesh = self._load_from_cache(self._cache_key)
            if cached_mesh is not None:
                self.mesh = cached_mesh
                # Apply normalization to cached mesh too
                if auto_normalize:
                    self._normalize_mesh(target_size_mm)
                self._filepath = filepath
                self._compute_info()
                logging.info("Using cached mesh data.")
                return self.to_scientific_data(source=filepath)
        
        # Load from file
        logging.info(f"Loading from {self._file_format.upper()} file (not cached)...")
        try:
            # trimesh auto-detects format from extension
            self.mesh = trimesh.load(filepath)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"Failed to load mesh file: {exc}") from exc
        
        # Ensure we have a Trimesh, not a Scene
        self.mesh = self._ensure_trimesh(self.mesh)
        
        # Final verification
        if isinstance(self.mesh, trimesh.Scene):
            raise ValueError("Failed to convert Scene to Mesh")
        
        # Auto-normalize mesh size
        if auto_normalize:
            self._normalize_mesh(target_size_mm)
            
        self._filepath = filepath
        self._compute_info()
        
        # Save to cache for future use
        if use_cache:
            self._save_to_cache()
        
        return self.to_scientific_data(source=filepath)

    def load_mesh(
        self,
        filepath: str | Path,
        use_cache: bool = True,
        auto_normalize: bool = True,
        target_size_mm: float = 100.0,
    ):
        """
        Backwards-compatible helper that returns raw trimesh.Trimesh.
        """
        data = self.load(
            filepath=filepath,
            use_cache=use_cache,
            auto_normalize=auto_normalize,
            target_size_mm=target_size_mm,
        )
        return data.primary_data

    def to_scientific_data(
        self,
        source: Optional[str | Path] = None,
    ) -> ScientificData[Any, dict[str, Any]]:
        """Package the currently loaded mesh into ScientificData."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")

        spatial_info = {}
        secondary_data = {}
        if self.info is not None:
            secondary_data["mesh_info"] = self.info
            spatial_info = {
                "bounds_min": self.info.bounds_min.copy(),
                "bounds_max": self.info.bounds_max.copy(),
                "dimensions_mm": self.info.dimensions.copy(),
                "center": self.info.center.copy(),
            }

        metadata = {
            "data_kind": "mesh",
            "file_format": self._file_format,
        }
        if self._cache_key:
            metadata["cache_key"] = self._cache_key
        if source is not None:
            metadata["source"] = str(source)
        elif self._filepath is not None:
            metadata["source"] = str(self._filepath)

        return ScientificData(
            primary_data=self.mesh,
            secondary_data=secondary_data,
            spatial_info=spatial_info,
            metadata=metadata,
        )
    
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
            file_format=self._file_format,
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


# Backwards compatibility alias
STLLoader = MeshLoader
