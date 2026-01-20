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
        physics_energy_bins: int = 10,
        voxel_grid=None,  # Optional pre-computed VoxelGrid
        structure_config=None  # Optional (method_name, config) for deferred generation
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
        # Pre-computed voxel grid (from StructurePanel)
        self.voxel_grid = voxel_grid
        # Deferred structure configuration
        self.structure_config = structure_config
    
    def run(self):
        try:
            timing_info = {
                'use_gpu': self.use_gpu,
                'fast_mode': self.fast_mode,
                'physics_mode': self.physics_mode,
                'voxelization_time': 0.0,
                'structure_time': 0.0,
                'simulation_time': 0.0,
                'total_time': 0.0,
                'gpu_timing': None
            }
            
            total_start = time.perf_counter()
            
            # Voxelization (skip if pre-computed grid provided)
            self.progress.emit(0.0)
            
            if self.voxel_grid is not None:
                # Use pre-computed voxel grid (from StructurePanel)
                # CLONE it to avoid modifying the version in DataManager (persistent defects)
                # since StructureModifier works in-place
                from simulation.voxelizer import VoxelGrid
                voxel_grid = VoxelGrid(
                    data=self.voxel_grid.data.copy(),
                    voxel_size=self.voxel_grid.voxel_size,
                    origin=self.voxel_grid.origin.copy()
                )
                timing_info['voxelization_time'] = 0.0
                logging.info("Using cloned pre-computed voxel grid from StructurePanel")
            else:
                # Voxelize from mesh
                vox_start = time.perf_counter()
                voxelizer = Voxelizer(
                    voxel_size=self.voxel_size,
                    fill_interior=self.fill_interior,
                    max_memory_gb=self.memory_limit_gb
                )
                voxel_grid = voxelizer.voxelize(self.mesh)
                timing_info['voxelization_time'] = time.perf_counter() - vox_start
            
            # Deferred Structure Generation
            if self.structure_config is not None:
                struct_start = time.perf_counter()
                self.progress.emit(0.1) # Start structure gen
                
                method_name, config = self.structure_config
                from simulation.structures import StructureModifier
                modifier = StructureModifier(voxel_grid)
                
                # Progress wrapper for structure generation
                def struct_progress(p):
                    # Map 0-1 to 0.1-0.3
                    self.progress.emit(0.1 + p * 0.2)
                
                logging.info(f"Applying deferred structure generation: {method_name}")
                if method_name == "generate_lattice":
                    modifier.generate_lattice(config, progress_callback=struct_progress)
                elif method_name == "generate_random_voids":
                    modifier.generate_random_voids(config, progress_callback=struct_progress)
                    
                timing_info['structure_time'] = time.perf_counter() - struct_start
                voxel_grid = modifier.grid
                
            self.progress.emit(0.3)  # Ready for sim
            
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


class StructureWorker(QThread):
    """Background worker for structure generation."""
    
    progress = Signal(float)
    finished = Signal(object)  # Emits VoxelGrid
    error = Signal(str)
    
    def __init__(
        self,
        modifier,  # Type: StructureModifier or None
        method_name: str,
        config: object,
        mesh=None,
        voxel_size: float = 0.5
    ):
        super().__init__()
        self.modifier = modifier
        self.method_name = method_name
        self.config = config
        self.mesh = mesh
        self.voxel_size = voxel_size
    
    def run(self):
        try:
            self.progress.emit(0.0)
            
            # Auto-initialization if needed
            start_progress = 0.0
            if self.modifier is None:
                if self.mesh is None:
                    raise ValueError("No modifier and no mesh provided")
                    
                from simulation.voxelizer import Voxelizer
                from simulation.structures import StructureModifier
                
                logging.info(f"Auto-initializing grid with voxel size {self.voxel_size}mm...")
                voxelizer = Voxelizer(voxel_size=self.voxel_size, fill_interior=True)
                voxel_grid = voxelizer.voxelize(self.mesh)
                self.modifier = StructureModifier(voxel_grid)
                
                self.progress.emit(0.3)  # Initialization done
                start_progress = 0.3
            
            # Progress callback wrapper
            # Scales 0.0-1.0 from modifier to start_progress-1.0
            def on_progress(p):
                # Ensure p is clamped 0-1
                p = max(0.0, min(1.0, float(p)))
                total_progress = start_progress + p * (1.0 - start_progress)
                self.progress.emit(total_progress)
            
            # Run generation with progress
            if self.method_name == "generate_lattice":
                self.modifier.generate_lattice(self.config, progress_callback=on_progress)
            elif self.method_name == "generate_random_voids":
                self.modifier.generate_random_voids(self.config, progress_callback=on_progress)
            
            self.progress.emit(1.0)
            self.finished.emit(self.modifier.grid)
            
        except Exception as e:
            import traceback
            logging.error(f"Structure generation error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))
