"""
Core Base Classes

Provides the fundamental data structures and abstract interfaces
for the scientific application following Data-Centric Architecture.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class ScientificData:
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
    primary_data: Any = None
    secondary_data: Any = None
    spatial_info: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def shape(self) -> Optional[Tuple]:
        """Get the shape of primary data if it's array-like."""
        if hasattr(self.primary_data, 'shape'):
            return self.primary_data.shape
        return None
    
    @property
    def dtype(self) -> Optional[np.dtype]:
        """Get the dtype of primary data if it's a numpy array."""
        if hasattr(self.primary_data, 'dtype'):
            return self.primary_data.dtype
        return None


class BaseLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, source: str) -> ScientificData:
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


class BaseAnalyzer(ABC):
    """Abstract base class for data analyzers/processors."""
    
    @abstractmethod
    def process(self, data: ScientificData, **params) -> ScientificData:
        """
        Process scientific data.
        
        Args:
            data: Input ScientificData
            **params: Processing parameters
            
        Returns:
            Processed ScientificData (may be new or modified)
        """
        pass


class BaseVisualizer(ABC):
    """Abstract base class for data visualizers."""
    
    @abstractmethod
    def set_data(self, data: ScientificData) -> None:
        """
        Set the data to visualize.
        
        Args:
            data: ScientificData to display
        """
        pass
    
    def clear(self) -> None:
        """Clear the visualization."""
        pass
