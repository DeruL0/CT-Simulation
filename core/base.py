"""
Core Base Classes

Provides the fundamental data structures and abstract interfaces
for the scientific application following Data-Centric Architecture.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Mapping, Optional, Tuple, Type, TypeVar
import numpy as np
import numpy.typing as npt
from .validation import require_ndarray as _require_ndarray


PrimaryDataT = TypeVar("PrimaryDataT")
SecondaryDataT = TypeVar("SecondaryDataT")

InPrimaryDataT = TypeVar("InPrimaryDataT")
InSecondaryDataT = TypeVar("InSecondaryDataT")
OutPrimaryDataT = TypeVar("OutPrimaryDataT")
OutSecondaryDataT = TypeVar("OutSecondaryDataT")
ProcessConfigT = TypeVar("ProcessConfigT")

RequiredT = TypeVar("RequiredT")
SecondaryRequiredT = TypeVar("SecondaryRequiredT")


@dataclass
class ScientificData(Generic[PrimaryDataT, SecondaryDataT]):
    """
    Generic DTO (Data Transfer Object) for scientific data.
    
    This serves as the single source of truth passed between
    Loaders, Analyzers, and Visualizers.
    
    Attributes:
        primary_data: The main array/tensor/mesh (e.g., voxel grid, CT volume)
        secondary_data: Auxiliary data (e.g., derived results, masks)
        spatial_info: Spacing, Origin, Units, Transform information
        metadata: Experiment ID, Timestamp, Processing history
    """
    primary_data: Optional[PrimaryDataT] = None
    secondary_data: Optional[SecondaryDataT] = None
    spatial_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get the shape of primary data if it's array-like."""
        value = self.primary_data
        if value is not None and hasattr(value, "shape"):
            return tuple(value.shape)
        return None
    
    @property
    def dtype(self) -> Optional[np.dtype[Any]]:
        """Get the dtype of primary data if it has one."""
        value = self.primary_data
        if value is not None and hasattr(value, "dtype"):
            return np.dtype(value.dtype)
        return None

    def require_primary(self, expected_type: Type[RequiredT]) -> RequiredT:
        """
        Assert primary_data type and return it.

        Raises:
            TypeError: if primary_data is missing or has unexpected type.
        """
        value = self.primary_data
        if value is None:
            raise TypeError("ScientificData.primary_data is None.")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"ScientificData.primary_data must be {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return value

    def require_secondary(self, expected_type: Type[SecondaryRequiredT]) -> SecondaryRequiredT:
        """
        Assert secondary_data type and return it.

        This makes `SecondaryDataT` a real runtime contract instead of
        a purely syntactic generic placeholder.

        Raises:
            TypeError: if secondary_data is missing or has unexpected type.
        """
        value = self.secondary_data
        if value is None:
            raise TypeError("ScientificData.secondary_data is None.")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"ScientificData.secondary_data must be {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return value

    def require_secondary_keys(self, required_keys: Iterable[str]) -> Mapping[str, Any]:
        """
        Assert secondary_data is a mapping and contains all required keys.

        Raises:
            TypeError: if secondary_data is not a mapping.
            KeyError: if required keys are missing.
        """
        mapping = self.require_secondary(Mapping)
        missing = [key for key in required_keys if key not in mapping]
        if missing:
            raise KeyError(
                f"ScientificData.secondary_data missing required keys: {missing}"
            )
        return mapping

    def require_ndarray(
        self,
        *,
        ndim: Optional[int] = None,
        dtype: Optional[npt.DTypeLike] = None,
        numeric: bool = False,
        require_c_contiguous: bool = False,
        coerce_c_contiguous: bool = False,
    ) -> npt.NDArray[Any]:
        """
        Assert primary_data is an ndarray and optionally validate ndim/dtype.

        Raises:
            TypeError: if primary_data is not a numpy array or dtype mismatches.
            ValueError: if ndim mismatches.
        """
        value = self.require_primary(np.ndarray)
        validated = _require_ndarray(
            value,
            name="ScientificData.primary_data",
            ndim=ndim,
            dtype=dtype,
            numeric=numeric,
            require_c_contiguous=require_c_contiguous,
            coerce_c_contiguous=coerce_c_contiguous,
        )
        if validated is not value:
            self.primary_data = validated  # type: ignore[assignment]
        return validated


@dataclass
class GPUSimulationTiming:
    """
    Strongly-typed GPU timing breakdown for CT simulation.
    """

    transfer_to_gpu: float = 0.0
    radon: float = 0.0
    iradon: float = 0.0
    transfer_to_cpu: float = 0.0
    total: float = 0.0
    slices: int = 0

    @staticmethod
    def _coerce_float(
        payload: Mapping[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        """Read numeric timing value from mapping with explicit fallback behavior."""
        value = payload.get(key, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(
        payload: Mapping[str, Any],
        key: str,
        default: int = 0,
    ) -> int:
        """Read integer timing value from mapping with explicit fallback behavior."""
        value = payload.get(key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GPUSimulationTiming":
        """Build from a raw mapping payload while coercing numeric values."""
        return cls(
            transfer_to_gpu=cls._coerce_float(payload, "transfer_to_gpu"),
            radon=cls._coerce_float(payload, "radon"),
            iradon=cls._coerce_float(payload, "iradon"),
            transfer_to_cpu=cls._coerce_float(payload, "transfer_to_cpu"),
            total=cls._coerce_float(payload, "total"),
            slices=cls._coerce_int(payload, "slices"),
        )


@dataclass
class SimulationTimingResult:
    """
    Strongly-typed simulation timing payload shared across worker/UI boundary.
    """

    use_gpu: bool
    fast_mode: bool
    physics_mode: bool

    voxelization_time: float = 0.0
    structure_time: float = 0.0
    simulation_time: float = 0.0
    total_time: float = 0.0

    compression_time: Optional[float] = None
    gpu_timing: Optional[GPUSimulationTiming] = None


class BaseLoader(ABC, Generic[PrimaryDataT, SecondaryDataT]):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, source: str) -> ScientificData[PrimaryDataT, SecondaryDataT]:
        """
        Load data from a source.
        
        Args:
            source: Path or URI to the data source
            
        Returns:
            ScientificData containing the loaded data
        """
        pass
    
    def can_load(self, source: str) -> bool:
        """
        Check if this loader can handle the given source.
        
        Args:
            source: Path or URI to check
            
        Returns:
            True if this loader can handle the source
        """
        return True


class BaseAnalyzer(
    ABC,
    Generic[
        InPrimaryDataT,
        InSecondaryDataT,
        OutPrimaryDataT,
        OutSecondaryDataT,
        ProcessConfigT,
    ],
):
    """Abstract base class for data analyzers/processors."""
    
    @abstractmethod
    def process(
        self,
        data: ScientificData[InPrimaryDataT, InSecondaryDataT],
        process_config: ProcessConfigT,
    ) -> ScientificData[OutPrimaryDataT, OutSecondaryDataT]:
        """
        Process scientific data.
        
        Args:
            data: Input ScientificData
            process_config: Strongly-typed processing configuration payload
            
        Returns:
            Processed ScientificData (may be new or modified)
        """
        pass


class BaseVisualizer(ABC, Generic[PrimaryDataT, SecondaryDataT]):
    """Abstract base class for data visualizers."""
    
    @abstractmethod
    def set_data(self, data: ScientificData[PrimaryDataT, SecondaryDataT]) -> None:
        """
        Set the data to visualize.
        
        Args:
            data: ScientificData to display
        """
        pass
    
    def clear(self) -> None:
        """Clear the visualization."""
        pass
