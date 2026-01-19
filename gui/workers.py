"""
Background Workers

QThread workers for long-running operations (loading, simulation, export).
"""

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from loaders.stl_loader import STLLoader
from simulation.voxelizer import Voxelizer
from simulation.ct_simulator import CTSimulator, CTVolume
from exporters.dicom import DICOMExporter


class LoaderWorker(QThread):
    """Background worker for loading STL files."""
    
    progress = Signal(float)
    finished = Signal(object)  # Emits STLLoader
    error = Signal(str)
    
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
    
    def run(self):
        try:
            self.progress.emit(0.0)
            
            loader = STLLoader()
            loader.load(self.filepath)
            
            self.progress.emit(1.0)
            self.finished.emit(loader)
            
        except Exception as e:
            import traceback
            logging.error(f"Loading error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class SimulationWorker(QThread):
    """Background worker for CT simulation."""
    
    progress = Signal(float)
    finished = Signal(object, dict)  # Emits (CTVolume, timing_info)
    error = Signal(str)
    
    def __init__(
        self,
        mesh,
        voxel_size: float,
        fill_interior: bool,
        num_projections: int,
        add_noise: bool,
        noise_level: float,
        material,
        fast_mode: bool,
        memory_limit_gb: float = 2.0,
        use_gpu: bool = False,
        physics_mode: bool = False,
        physics_kvp: int = 120,
        physics_tube_current: int = 200,
        physics_filtration: float = 2.5,
        physics_energy_bins: int = 10
    ):
        super().__init__()
        self.mesh = mesh
        self.voxel_size = voxel_size
        self.fill_interior = fill_interior
        self.num_projections = num_projections
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.material = material
        self.fast_mode = fast_mode
        self.memory_limit_gb = memory_limit_gb
        self.use_gpu = use_gpu
        # Physics mode
        self.physics_mode = physics_mode
        self.physics_kvp = physics_kvp
        self.physics_tube_current = physics_tube_current
        self.physics_filtration = physics_filtration
        self.physics_energy_bins = physics_energy_bins
    
    def run(self):
        try:
            timing_info = {
                'use_gpu': self.use_gpu,
                'fast_mode': self.fast_mode,
                'physics_mode': self.physics_mode,
                'voxelization_time': 0.0,
                'simulation_time': 0.0,
                'total_time': 0.0,
                'gpu_timing': None
            }
            
            total_start = time.perf_counter()
            
            # Voxelization
            self.progress.emit(0.0)
            vox_start = time.perf_counter()
            voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                fill_interior=self.fill_interior,
                max_memory_gb=self.memory_limit_gb
            )
            voxel_grid = voxelizer.voxelize(self.mesh)
            timing_info['voxelization_time'] = time.perf_counter() - vox_start
            self.progress.emit(0.2)
            
            # CT Simulation
            sim_start = time.perf_counter()
            
            def sim_progress(p):
                self.progress.emit(0.2 + p * 0.8)
            
            if self.physics_mode:
                # Use physical simulation with polychromatic physics
                from simulation.physics import PhysicalCTSimulator, PhysicsConfig
                
                physics_config = PhysicsConfig(
                    kvp=self.physics_kvp,
                    tube_current_ma=self.physics_tube_current,
                    filtration_mm_al=self.physics_filtration,
                    energy_bins=self.physics_energy_bins,
                    use_gpu=self.use_gpu
                )
                
                simulator = PhysicalCTSimulator(
                    config=physics_config,
                    num_projections=self.num_projections
                )
                
                ct_volume = simulator.simulate(
                    voxel_grid,
                    material=self.material,
                    progress_callback=sim_progress
                )
                
            elif self.fast_mode:
                simulator = CTSimulator(
                    num_projections=self.num_projections,
                    add_noise=self.add_noise,
                    noise_level=self.noise_level,
                    use_gpu=self.use_gpu
                )
                ct_volume = simulator.simulate_fast(
                    voxel_grid,
                    material=self.material
                )
            else:
                simulator = CTSimulator(
                    num_projections=self.num_projections,
                    add_noise=self.add_noise,
                    noise_level=self.noise_level,
                    use_gpu=self.use_gpu
                )
                ct_volume = simulator.simulate(
                    voxel_grid,
                    material=self.material,
                    progress_callback=sim_progress
                )
            
            timing_info['simulation_time'] = time.perf_counter() - sim_start
            timing_info['total_time'] = time.perf_counter() - total_start
            
            # Get GPU-specific timing if available
            if hasattr(simulator, '_last_gpu_timing'):
                timing_info['gpu_timing'] = simulator._last_gpu_timing
            
            self.progress.emit(1.0)
            self.finished.emit(ct_volume, timing_info)
            
        except Exception as e:
            import traceback
            logging.error(f"Simulation error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class ExportWorker(QThread):
    """Background worker for DICOM export."""
    
    progress = Signal(float)
    finished = Signal(list)  # Emits list of file paths
    error = Signal(str)
    
    def __init__(
        self,
        ct_volume: CTVolume,
        output_dir: str,
        window_center: float,
        window_width: float
    ):
        super().__init__()
        self.ct_volume = ct_volume
        self.output_dir = output_dir
        self.window_center = window_center
        self.window_width = window_width
    
    def run(self):
        try:
            exporter = DICOMExporter()
            
            files = exporter.export(
                self.ct_volume,
                self.output_dir,
                window_center=self.window_center,
                window_width=self.window_width,
                progress_callback=self.progress.emit
            )
            
            self.finished.emit(files)
            
        except Exception as e:
            import traceback
            logging.error(f"Export error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))
