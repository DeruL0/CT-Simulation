"""
Data Manager

Centralized GUI-layer data state management for the application.
"""

import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from core import ScientificData
from loaders import MeshLoader as STLLoader
from simulation.volume import CTVolume
from simulation.voxelizer import VoxelGrid


class DataManager(QObject):
    """
    Manages application data state.

    Provides a centralized location for:
    - Mesh data
    - Voxel grid (modified for structures)
    - CT volume data
    - State change notifications via signals
    """

    stl_loaded = Signal(object)  # Emits STLLoader
    voxel_grid_changed = Signal(object)  # Emits VoxelGrid or None
    ct_volume_changed = Signal(object)  # Emits CTVolume or None

    def __init__(self, parent=None):
        super().__init__(parent)

        self._stl_loader: Optional[STLLoader] = None
        self._mesh_data: Optional[ScientificData[Any, dict[str, Any]]] = None

        self._voxel_grid: Optional[VoxelGrid] = None
        self._voxel_data: Optional[ScientificData[VoxelGrid, dict[str, Any]]] = None

        self._ct_volume: Optional[CTVolume] = None
        self._ct_data: Optional[ScientificData[CTVolume, dict[str, Any]]] = None

    @property
    def stl_loader(self) -> Optional[STLLoader]:
        """Current mesh loader."""
        return self._stl_loader

    @property
    def mesh_data(self) -> Optional[ScientificData[Any, dict[str, Any]]]:
        """Current mesh ScientificData payload."""
        return self._mesh_data

    @property
    def mesh(self):
        """Current mesh from ScientificData (or loader fallback)."""
        if self._mesh_data is not None and self._mesh_data.primary_data is not None:
            return self._mesh_data.primary_data
        if self._stl_loader is not None:
            return self._stl_loader.mesh
        return None

    @property
    def stl_info(self):
        """Current mesh info (loader preferred, ScientificData fallback)."""
        if self._stl_loader is not None:
            return self._stl_loader.info
        if self._mesh_data is not None and isinstance(self._mesh_data.secondary_data, dict):
            return self._mesh_data.secondary_data.get("mesh_info")
        return None

    @property
    def voxel_grid(self) -> Optional[VoxelGrid]:
        """Current voxel grid (may be modified with structures)."""
        return self._voxel_grid

    @property
    def voxel_data(self) -> Optional[ScientificData[VoxelGrid, dict[str, Any]]]:
        """Current voxel ScientificData payload."""
        return self._voxel_data

    @property
    def ct_volume(self) -> Optional[CTVolume]:
        """Current CT volume."""
        return self._ct_volume

    @property
    def ct_data(self) -> Optional[ScientificData[CTVolume, dict[str, Any]]]:
        """Current CT ScientificData payload."""
        return self._ct_data

    @property
    def has_stl(self) -> bool:
        """Whether mesh data is loaded."""
        return self.mesh is not None

    @property
    def has_voxel_grid(self) -> bool:
        """Whether voxel grid is available."""
        return self._voxel_grid is not None

    @property
    def has_ct_volume(self) -> bool:
        """Whether CT volume is available."""
        return self._ct_volume is not None

    def load_stl(self, filepath: str) -> bool:
        """
        Load a mesh file.

        Args:
            filepath: Path to mesh file

        Returns:
            True if loaded successfully
        """
        try:
            loader = STLLoader()
            mesh_data = loader.load(filepath)
            self.set_mesh_data(mesh_data, stl_loader=loader)
            logging.info("Loaded mesh: %s", filepath)
            return True
        except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
            logging.error("Failed to load mesh: %s", exc)
            return False

    def set_stl_loader(self, stl_loader: Optional[STLLoader]) -> None:
        """
        Set the current loader and reset derived data.

        Args:
            stl_loader: Mesh loader instance or None to clear
        """
        if stl_loader is None:
            self.set_mesh_data(None, stl_loader=None)
            return

        mesh_data = stl_loader.to_scientific_data(source=stl_loader.filepath)
        self.set_mesh_data(mesh_data, stl_loader=stl_loader)

    def set_mesh_data(
        self,
        mesh_data: Optional[ScientificData[Any, dict[str, Any]]],
        stl_loader: Optional[STLLoader] = None,
    ) -> None:
        """
        Set current mesh payload and reset derived data.

        Args:
            mesh_data: ScientificData containing mesh as primary_data
            stl_loader: Optional loader reference for legacy UI access
        """
        if mesh_data is not None and not isinstance(mesh_data, ScientificData):
            raise TypeError(
                f"mesh_data must be ScientificData or None, got {type(mesh_data).__name__}"
            )

        self._mesh_data = mesh_data
        self._stl_loader = stl_loader
        self._voxel_grid = None
        self._voxel_data = None
        self._ct_volume = None
        self._ct_data = None

        stl_payload = self._stl_loader if self._stl_loader is not None else mesh_data
        self.stl_loaded.emit(stl_payload)
        self.voxel_grid_changed.emit(None)
        self.ct_volume_changed.emit(None)

        if mesh_data is not None:
            source = mesh_data.metadata.get("source", "<unknown>")
            logging.info("Mesh data set from source: %s", source)

    def set_voxel_grid(self, voxel_grid: Optional[VoxelGrid]) -> None:
        """
        Set the current voxel grid.

        Args:
            voxel_grid: VoxelGrid instance or None to clear
        """
        self._voxel_grid = voxel_grid
        if voxel_grid is None:
            self._voxel_data = None
        else:
            self._voxel_data = ScientificData(
                primary_data=voxel_grid,
                secondary_data={},
                spatial_info={
                    "voxel_size_mm": voxel_grid.voxel_size,
                    "origin": voxel_grid.origin.copy(),
                },
                metadata={"stage": "voxelized"},
            )
        self.voxel_grid_changed.emit(voxel_grid)
        if voxel_grid is not None:
            logging.info("Voxel grid set: %s", voxel_grid.shape)

    def set_voxel_data(
        self,
        voxel_data: Optional[ScientificData[VoxelGrid, dict[str, Any]]],
    ) -> None:
        """
        Set voxel ScientificData payload and update legacy voxel_grid field.
        """
        if voxel_data is not None and not isinstance(voxel_data, ScientificData):
            raise TypeError(
                f"voxel_data must be ScientificData or None, got {type(voxel_data).__name__}"
            )

        self._voxel_data = voxel_data
        payload = voxel_data.primary_data if voxel_data is not None else None
        self._voxel_grid = payload if isinstance(payload, VoxelGrid) else None
        self.voxel_grid_changed.emit(self._voxel_grid)
        if self._voxel_grid is not None:
            logging.info("Voxel data set: %s", self._voxel_grid.shape)

    def clear_voxel_grid(self) -> None:
        """Clear the current voxel grid."""
        self.set_voxel_grid(None)

    def set_ct_volume(self, ct_volume: Optional[CTVolume]) -> None:
        """
        Set the current CT volume.

        Args:
            ct_volume: CTVolume instance or None to clear
        """
        self._ct_volume = ct_volume
        if ct_volume is None:
            self._ct_data = None
        else:
            self._ct_data = ScientificData(
                primary_data=ct_volume,
                secondary_data={},
                spatial_info={
                    "voxel_size_mm": ct_volume.voxel_size,
                    "origin": ct_volume.origin.copy(),
                },
                metadata={"stage": "simulated"},
            )
        self.ct_volume_changed.emit(ct_volume)
        if ct_volume is not None:
            logging.info("CT volume set: %s", ct_volume.shape)

    def set_ct_data(
        self,
        ct_data: Optional[ScientificData[CTVolume, dict[str, Any]]],
    ) -> None:
        """
        Set CT ScientificData payload and update legacy ct_volume field.
        """
        if ct_data is not None and not isinstance(ct_data, ScientificData):
            raise TypeError(
                f"ct_data must be ScientificData or None, got {type(ct_data).__name__}"
            )

        self._ct_data = ct_data
        payload = ct_data.primary_data if ct_data is not None else None
        self._ct_volume = payload if isinstance(payload, CTVolume) else None
        self.ct_volume_changed.emit(self._ct_volume)
        if self._ct_volume is not None:
            logging.info("CT data set: %s", self._ct_volume.shape)

    def clear_ct_volume(self) -> None:
        """Clear the current CT volume."""
        self.set_ct_volume(None)

    def clear(self) -> None:
        """Clear all data."""
        self._stl_loader = None
        self._mesh_data = None
        self._voxel_grid = None
        self._voxel_data = None
        self._ct_volume = None
        self._ct_data = None
        self.voxel_grid_changed.emit(None)
        self.ct_volume_changed.emit(None)
        logging.info("Data cleared")
