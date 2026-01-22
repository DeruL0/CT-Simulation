"""
Background Workers

QThread workers for long-running operations (loading, simulation, export).
"""

import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np

from PySide6.QtCore import QThread, Signal

from loaders import MeshLoader as STLLoader
from simulation.voxelizer import Voxelizer, VoxelGrid
from simulation.volume import CTVolume
from simulation.backends import get_backend
from simulation.simple_simulator import SimpleCTSimulator
from exporters.dicom import DICOMExporter
from .progress import TaskProgressTracker, ProgressPhase, get_standard_phases, get_compression_phases



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
    finished = Signal(object, dict, list)  # Emits (CTVolume, timing_info, compression_results)
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
        structure_config=None,  # Optional (method_name, config) for deferred generation
        compression_config=None  # Optional dict from CompressionPanel.get_config()
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
        # Compression configuration
        self.compression_config = compression_config
    
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
            
            has_compression = self.compression_config and self.compression_config.get('enabled')
            
            # Setup progress tracker with appropriate phases
            tracker = TaskProgressTracker(emit_fn=self.progress.emit)
            if has_compression:
                tracker.set_phases(get_compression_phases())
            else:
                tracker.set_phases(get_standard_phases())

            # ===== PHASE 0: Voxelization =====
            logging.info("=" * 50)
            logging.info("PHASE 1: Voxelization")
            tracker.start_phase(0)
            
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
                logging.info(f"  Using cloned pre-computed voxel grid: {voxel_grid.shape}")
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
                logging.info(f"  Voxelized: {voxel_grid.shape} in {timing_info['voxelization_time']:.2f}s")
            
            tracker.end_phase()  # End voxelization phase

            # ===== PHASE 1: Deferred Structure Generation =====
            if self.structure_config is not None:
                logging.info("=" * 50)
                logging.info("PHASE 2: Structure Generation")
                tracker.start_phase(1)
                struct_start = time.perf_counter()
                
                method_name, config = self.structure_config
                from simulation.structures import StructureModifier
                modifier = StructureModifier(voxel_grid)
                
                # Progress wrapper for structure generation with logging
                struct_cb = tracker.sub_progress(1)
                last_log_time = [time.perf_counter()]
                def struct_progress(p):
                    struct_cb(p)
                    now = time.perf_counter()
                    if now - last_log_time[0] > 2.0:
                        logging.info(f"  Structure progress: {p*100:.1f}%")
                        last_log_time[0] = now
                
                logging.info(f"  Method: {method_name}")
                logging.info(f"  preserve_shell={getattr(config, 'preserve_shell', False)}, "
                            f"shell_thickness={getattr(config, 'shell_thickness_mm', 0)}mm")
                
                if method_name == "generate_lattice":
                    modifier.generate_lattice(config, progress_callback=struct_progress)
                elif method_name == "generate_random_voids":
                    modifier.generate_random_voids(config, progress_callback=struct_progress)
                    
                timing_info['structure_time'] = time.perf_counter() - struct_start
                logging.info(f"  Structure generation completed in {timing_info['structure_time']:.2f}s")
                voxel_grid = modifier.grid
                tracker.end_phase()  # End structure phase
            
            # ===== PHASE 2 (or 1 if no structure): CT Simulation =====
            ct_phase = 2 if has_compression else 2  # Phase index for CT sim
            logging.info("=" * 50)
            logging.info(f"PHASE 3: {'Initial ' if has_compression else ''}CT Simulation")
            tracker.start_phase(ct_phase)
            sim_start = time.perf_counter()
            
            sim_cb = tracker.sub_progress(ct_phase)
            last_sim_log = [time.perf_counter()]
            def sim_progress(p):
                sim_cb(p)
                now = time.perf_counter()
                if now - last_sim_log[0] > 2.0:
                    logging.info(f"  Simulation progress: {p*100:.1f}%")
                    last_sim_log[0] = now
            
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
                
            else:
                # Normal mode or fast mode: use SimpleCTSimulator
                # (Fast mode skips reconstruction in SimpleCTSimulator if applicable)
                simulator = SimpleCTSimulator(
                    num_projections=self.num_projections,
                    use_gpu=self.use_gpu
                )
                ct_volume = simulator.simulate(
                    voxel_grid,
                    material=self.material,
                    progress_callback=sim_progress
                )
            
            timing_info['simulation_time'] = time.perf_counter() - sim_start
            tracker.end_phase()  # End CT simulation phase
            
            # Get GPU-specific timing if available
            if hasattr(simulator, '_last_gpu_timing'):
                timing_info['gpu_timing'] = simulator._last_gpu_timing
            
            # ===== PHASE 3-4: Compression Simulation (only if enabled) =====
            compression_results = []
            if has_compression:
                logging.info("=" * 50)
                logging.info("PHASE 4: Compression Simulation workflow")
                tracker.start_phase(3)  # Compression Physics phase
                comp_start = time.perf_counter()
                
                from simulation.mechanics import CompressionManager
                from simulation.mechanics.manager import CompressionConfig, CompressionResult
                
                cfg = self.compression_config
                manager = CompressionManager(use_gpu=cfg.get('use_gpu', True))
                
                comp_config = CompressionConfig(
                    total_compression=cfg['total_compression'],
                    num_steps=cfg['num_steps'],
                    poisson_ratio=cfg['poisson_ratio'],
                    axis=cfg.get('axis', 'Z'),
                    use_physics=cfg['mode'] == 'physical',
                    downsample_factor=cfg.get('downsample_factor', 4),
                    solver_iterations=cfg.get('solver_iterations', 300),
                    use_gpu=cfg.get('use_gpu', True),
                    export_dicom=False
                )
                
                # First, run compression on VOXEL GRID (binary data), not CT volume
                voxel_data = voxel_grid.data.astype(np.float32)
                
                # Get deformed voxel grids for each step
                phys_cb = tracker.sub_progress(3)
                def phys_progress(p, s):
                    phys_cb(p)

                deformed_grids = manager.run_simulation(
                    voxel_data,
                    voxel_grid.voxel_size,
                    comp_config,
                    progress_callback=phys_progress
                )
                tracker.end_phase()  # End physics phase
                
                logging.info(f"  Generated {len(deformed_grids)} compressed voxel grids")
                
                # Now run CT simulation for each compressed voxel grid
                from simulation.voxelizer import VoxelGrid
                
                tracker.start_phase(4)  # Batch CT phase
                num_steps = len(deformed_grids)
                for i, deform_result in enumerate(deformed_grids):
                    # Map loop progress within Batch CT phase
                    step_cb = tracker.sub_range(i / num_steps, (i + 1) / num_steps, 4)
                    
                    # OPTIMIZATION: If compression ratio is 0.0 (Step 0), reuse Phase 3 result
                    # Only valid if Phase 3 ran (which it does) and is uncompressed
                    if deform_result.compression_ratio == 0.0 and ct_volume is not None:
                         logging.info(f"  Skipping simulation for Step {i} (0% compression) - reusing initial CT")
                         step_ct_data = ct_volume.data
                         step_ct_voxel_size = ct_volume.voxel_size
                         step_cb(1.0) # Complete progress for this step instantly
                    else:
                        logging.info(f"  CT simulation for step {i}/{num_steps-1}...")
                        
                        # Create VoxelGrid from deformed data
                        deformed_grid = VoxelGrid(
                            data=deform_result.volume,
                            voxel_size=deform_result.voxel_size,
                            origin=voxel_grid.origin
                        )
                        
                        # Run CT simulation on this deformed grid
                        step_ct = simulator.simulate(
                            deformed_grid,
                            material=self.material,
                            progress_callback=step_cb
                        )
                        step_ct_data = step_ct.data
                        step_ct_voxel_size = step_ct.voxel_size
                    
                    # Store as CompressionResult with CT data
                    compression_results.append(CompressionResult(
                        step_index=deform_result.step_index,
                        compression_ratio=deform_result.compression_ratio,
                        _volume=step_ct_data,  # Use CT simulated data
                        voxel_size=step_ct_voxel_size
                    ))
                
                timing_info['compression_time'] = time.perf_counter() - comp_start
                logging.info(f"  Compression + CT: {len(compression_results)} steps in {timing_info['compression_time']:.2f}s")
            
            timing_info['total_time'] = time.perf_counter() - total_start
            
            # ===== Summary =====
            logging.info("=" * 50)
            logging.info("TIMING SUMMARY:")
            logging.info(f"  Voxelization:     {timing_info['voxelization_time']:.2f}s")
            logging.info(f"  Structure Gen:    {timing_info['structure_time']:.2f}s")
            logging.info(f"  CT Simulation:    {timing_info['simulation_time']:.2f}s")
            if timing_info.get('compression_time'):
                logging.info(f"  Compression:      {timing_info['compression_time']:.2f}s")
            logging.info(f"  TOTAL:            {timing_info['total_time']:.2f}s")
            if timing_info.get('gpu_timing'):
                logging.info(f"  GPU Details: {timing_info['gpu_timing']}")
            logging.info("=" * 50)
            
            self.progress.emit(1.0)
            self.finished.emit(ct_volume, timing_info, compression_results)
            
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
        volume_or_list,  # CTVolume or list of CTVolume/CompressionResult
        output_dir: str,
        window_center: float,
        window_width: float
    ):
        super().__init__()
        self.volume_or_list = volume_or_list
        self.output_dir = output_dir
        self.window_center = window_center
        self.window_width = window_width
    
    def run(self):
        try:
            exporter = DICOMExporter()
            all_files = []
            
            # Check if input is a list (time series)
            if isinstance(self.volume_or_list, list):
                items = self.volume_or_list
                num_items = len(items)
                
                for i, item in enumerate(items):
                    # Handle both CompressionResult objects and CTVolume/arrays
                    if hasattr(item, 'volume'):
                        # It's a CompressionResult
                        vol_data = item.volume
                        voxel_size = item.voxel_size
                        origin = getattr(item, 'origin', np.zeros(3))
                    elif hasattr(item, 'data'):
                        # It's a CTVolume
                        vol_data = item.data
                        voxel_size = item.voxel_size
                        origin = item.origin
                    else:
                        # Assume numpy array
                        vol_data = item
                        voxel_size = 0.5  # Default fallback
                        origin = np.zeros(3)
                        
                    # Create subdirectory (parents=True for nested paths)
                    step_dir = Path(self.output_dir) / f"Step_{i:02d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create temporary CTVolume wrapper for exporter
                    from simulation.volume import CTVolume
                    temp_vol = CTVolume(vol_data, voxel_size, origin)
                    
                    # Progress callback for this step - capture i in closure
                    step_idx = i  # Capture current i
                    def step_progress(p, idx=step_idx):
                        # Map 0-1 to global progress
                        global_p = (idx + p) / num_items
                        self.progress.emit(global_p)
                    
                    files = exporter.export(
                        temp_vol,
                        str(step_dir),
                        window_center=self.window_center,
                        window_width=self.window_width,
                        progress_callback=step_progress
                    )
                    all_files.extend(files)
                    
            else:
                # Single volume
                files = exporter.export(
                    self.volume_or_list,
                    self.output_dir,
                    window_center=self.window_center,
                    window_width=self.window_width,
                    progress_callback=self.progress.emit
                )
                all_files.extend(files)
            
            self.finished.emit(all_files)
            
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
