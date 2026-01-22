"""
Progress Tracking Utilities

Provides a flexible progress tracking system for multi-phase workflows.
Handles automatic progress scaling and sub-task progress mapping.
"""

from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ProgressPhase:
    """Defines a phase in a workflow with its progress weight."""
    name: str
    weight: float  # Relative weight (will be normalized)
    
    
class TaskProgressTracker:
    """
    Manages progress tracking for multi-phase workflows.
    
    Automatically calculates progress milestones based on phase weights
    and provides sub-progress callbacks that map to the correct range.
    
    Example:
        tracker = TaskProgressTracker(emit_fn=self.progress.emit)
        tracker.set_phases([
            ProgressPhase("Voxelization", 1),
            ProgressPhase("Structure", 2),
            ProgressPhase("Simulation", 7),
        ])
        
        # Phase 0: Voxelization (0.0 -> 0.1)
        tracker.start_phase(0)
        do_voxelization(progress_callback=tracker.sub_progress())
        
        # Phase 1: Structure (0.1 -> 0.3)
        tracker.start_phase(1)
        do_structure(progress_callback=tracker.sub_progress())
        
        # Phase 2: Simulation (0.3 -> 1.0)
        tracker.start_phase(2)
        do_simulation(progress_callback=tracker.sub_progress())
    """
    
    def __init__(self, emit_fn: Callable[[float], None]):
        """
        Initialize tracker.
        
        Args:
            emit_fn: Function to call with progress value (0.0 - 1.0)
        """
        self._emit = emit_fn
        self._phases: List[ProgressPhase] = []
        self._milestones: List[float] = []  # Start points for each phase
        self._current_phase: int = -1
        
    def set_phases(self, phases: List[ProgressPhase]) -> None:
        """
        Set the workflow phases.
        
        Args:
            phases: List of ProgressPhase objects defining the workflow
        """
        self._phases = phases
        
        # Normalize weights and calculate milestones
        total_weight = sum(p.weight for p in phases)
        if total_weight <= 0:
            total_weight = 1.0
            
        cumulative = 0.0
        self._milestones = []
        for phase in phases:
            self._milestones.append(cumulative)
            cumulative += phase.weight / total_weight
        self._milestones.append(1.0)  # End milestone
        
    def get_range(self, phase_index: int) -> Tuple[float, float]:
        """
        Get the progress range for a phase.
        
        Args:
            phase_index: Index of the phase
            
        Returns:
            Tuple of (start, end) progress values
        """
        if phase_index < 0 or phase_index >= len(self._phases):
            return (0.0, 1.0)
        return (self._milestones[phase_index], self._milestones[phase_index + 1])
    
    def start_phase(self, phase_index: int) -> None:
        """
        Start a new phase and emit its starting progress.
        
        Args:
            phase_index: Index of the phase to start
        """
        self._current_phase = phase_index
        start, _ = self.get_range(phase_index)
        self._emit(start)
        
    def end_phase(self) -> None:
        """End the current phase and emit its ending progress."""
        if self._current_phase >= 0:
            _, end = self.get_range(self._current_phase)
            self._emit(end)
    
    def sub_progress(self, phase_index: Optional[int] = None) -> Callable[[float], None]:
        """
        Create a sub-progress callback for a phase.
        
        The returned callback maps 0.0-1.0 to the phase's progress range.
        
        Args:
            phase_index: Phase index (defaults to current phase)
            
        Returns:
            Callback function that accepts progress 0.0-1.0
        """
        if phase_index is None:
            phase_index = self._current_phase
            
        start, end = self.get_range(phase_index)
        
        def callback(p: float) -> None:
            # Clamp to 0-1
            p = max(0.0, min(1.0, p))
            self._emit(start + p * (end - start))
            
        return callback
    
    def sub_range(
        self, 
        sub_start: float, 
        sub_end: float,
        phase_index: Optional[int] = None
    ) -> Callable[[float], None]:
        """
        Create a sub-progress callback for a portion of a phase.
        
        Useful for dividing a phase into multiple sub-tasks.
        
        Args:
            sub_start: Start of sub-range within phase (0.0-1.0)
            sub_end: End of sub-range within phase (0.0-1.0)
            phase_index: Phase index (defaults to current phase)
            
        Returns:
            Callback function that accepts progress 0.0-1.0
        """
        if phase_index is None:
            phase_index = self._current_phase
            
        phase_start, phase_end = self.get_range(phase_index)
        phase_len = phase_end - phase_start
        
        # Map sub-range to actual progress range
        actual_start = phase_start + sub_start * phase_len
        actual_end = phase_start + sub_end * phase_len
        
        def callback(p: float) -> None:
            p = max(0.0, min(1.0, p))
            self._emit(actual_start + p * (actual_end - actual_start))
            
        return callback
    
    def emit(self, progress: float) -> None:
        """Directly emit a progress value."""
        self._emit(max(0.0, min(1.0, progress)))
        
    @property
    def current_phase_name(self) -> str:
        """Get the name of the current phase."""
        if 0 <= self._current_phase < len(self._phases):
            return self._phases[self._current_phase].name
        return ""


# Preset phase configurations for common workflows
def get_standard_phases() -> List[ProgressPhase]:
    """Standard CT simulation workflow phases."""
    return [
        ProgressPhase("Voxelization", 1),
        ProgressPhase("Structure", 2),
        ProgressPhase("Simulation", 7),
    ]


def get_compression_phases() -> List[ProgressPhase]:
    """Compression workflow phases."""
    return [
        ProgressPhase("Voxelization", 1),
        ProgressPhase("Structure", 2),
        ProgressPhase("Initial CT", 2),
        ProgressPhase("Compression Physics", 3),
        ProgressPhase("Batch CT", 12),
    ]
