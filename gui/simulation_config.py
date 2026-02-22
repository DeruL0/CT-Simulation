"""
Simulation configuration DTOs and builder.

Provides GUI-agnostic configuration records used by orchestration logic.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

from simulation.materials import MaterialType


StructureConfig = Optional[Tuple[str, Any]]
CompressionConfig = Optional[dict[str, Any]]


@dataclass(frozen=True)
class CTSimulationConfig:
    """
    Immutable simulation configuration record.

    This DTO intentionally carries no Qt/UI widget references.
    """

    voxel_size: float
    fill_interior: bool
    num_projections: int
    add_noise: bool
    noise_level: float
    material: MaterialType
    fast_mode: bool
    memory_limit_gb: float
    use_gpu: bool
    physics_mode: bool
    physics_kvp: int
    physics_tube_current: int
    physics_filtration: float
    physics_energy_bins: int
    physics_exposure_multiplier: float
    structure_config: StructureConfig
    compression_config: CompressionConfig


class _ParamsPanelLike(Protocol):
    def build_simulation_config(
        self,
        *,
        material: MaterialType,
        structure_config: StructureConfig,
        compression_config: CompressionConfig,
    ) -> CTSimulationConfig:
        ...


class _LoaderPanelLike(Protocol):
    @property
    def selected_material(self) -> MaterialType:
        ...


class _StructurePanelLike(Protocol):
    def build_structure_config(self) -> StructureConfig:
        ...


class _CompressionPanelLike(Protocol):
    def build_compression_config(self) -> CompressionConfig:
        ...


class SimulationConfigBuilder:
    """Builds `CTSimulationConfig` from panel-level abstractions."""

    def __init__(
        self,
        params_panel: _ParamsPanelLike,
        loader_panel: _LoaderPanelLike,
        structure_panel: _StructurePanelLike,
        compression_panel: _CompressionPanelLike,
    ):
        self._params_panel = params_panel
        self._loader_panel = loader_panel
        self._structure_panel = structure_panel
        self._compression_panel = compression_panel

    def build(self) -> CTSimulationConfig:
        """Assemble the simulation DTO without exposing widget internals."""
        structure_config = self._structure_panel.build_structure_config()
        compression_config = self._compression_panel.build_compression_config()
        return self._params_panel.build_simulation_config(
            material=self._loader_panel.selected_material,
            structure_config=structure_config,
            compression_config=compression_config,
        )
