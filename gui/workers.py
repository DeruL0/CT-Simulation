"""
Background Workers

QThread workers for long-running operations (loading, simulation, export).
"""

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from core import ScientificData, GPUSimulationTiming, SimulationTimingResult
from loaders import MeshLoader as STLLoader
from simulation.materials import MaterialType
from simulation.voxelizer import Voxelizer, VoxelGrid
from simulation.volume import CTVolume
from simulation.simple_simulator import SimpleCTSimulator, SimpleProcessConfig
from .progress import TaskProgressTracker, get_standard_phases, get_compression_phases
from .compression_pipeline import run_compression_workflow
from .export_pipeline import export_volume_or_series



class LoaderWorker(QThread):
    """Background worker for loading STL files."""
    
    progress = Signal(float)
    finished = Signal(object)  # Emits ScientificData
    error = Signal(str)
    
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
    
    def run(self):
        try:
            self.progress.emit(0.0)
            
            loader = STLLoader()
            mesh_data = loader.load(self.filepath)
            if isinstance(mesh_data.secondary_data, dict):
                mesh_data.secondary_data["loader"] = loader
            
            self.progress.emit(1.0)
            self.finished.emit(mesh_data)
            
        except (FileNotFoundError, ValueError, TypeError, OSError, ImportError, RuntimeError) as e:
            import traceback
            logging.error(f"Loading error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class SimulationWorker(QThread):
    """Background worker for CT simulation."""
    
    progress = Signal(float)
    finished = Signal(object, object, list, object)  # (ScientificData[CTVolume], SimulationTimingResult, compression_results, annotations)
    error = Signal(str)
    
    def __init__(
        self,
        mesh,
        voxel_size: float,
        fill_interior: bool,
        num_projections: int,
        add_noise: bool,
        noise_level: float,
        material: MaterialType,
        fast_mode: bool,
        memory_limit_gb: float = 2.0,
        use_gpu: bool = False,
        physics_mode: bool = False,
        physics_kvp: int = 120,
        physics_tube_current: int = 200,
        physics_filtration: float = 2.5,
        physics_energy_bins: int = 10,
        physics_exposure_multiplier: float = 1.0,
        voxel_grid=None,  # Optional pre-computed VoxelGrid
        mesh_data: Optional[ScientificData] = None,  # Optional mesh ScientificData input
        voxel_data: Optional[ScientificData] = None,  # Optional voxel ScientificData input
        structure_config=None,  # Optional (method_name, config) for deferred generation
        compression_config=None  # Optional dict from CompressionPanel.get_config()
    ):
        super().__init__()
        self.mesh = mesh
        self.mesh_data = mesh_data
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
        self.physics_exposure_multiplier = physics_exposure_multiplier
        # Pre-computed voxel grid (from StructurePanel)
        self.voxel_grid = voxel_grid
        # ScientificData voxel payload
        self.voxel_data = voxel_data
        # Deferred structure configuration
        self.structure_config = structure_config
        # Compression configuration
        self.compression_config = compression_config
    
    def run(self):
        try:
            timing_info = SimulationTimingResult(
                use_gpu=self.use_gpu,
                fast_mode=self.fast_mode,
                physics_mode=self.physics_mode,
            )
            
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

            mesh = self.mesh
            if isinstance(self.mesh_data, ScientificData) and self.mesh_data.primary_data is not None:
                mesh = self.mesh_data.primary_data

            input_voxel_grid = self.voxel_grid
            if isinstance(self.voxel_data, ScientificData):
                input_voxel_grid = self.voxel_data.primary_data
            elif isinstance(self.voxel_data, VoxelGrid):
                input_voxel_grid = self.voxel_data
            
            if input_voxel_grid is not None:
                # Use pre-computed voxel grid (from StructurePanel)
                # CLONE it to avoid modifying the version in DataManager (persistent defects)
                # since StructureModifier works in-place
                voxel_grid = VoxelGrid(
                    data=input_voxel_grid.data.copy(),
                    voxel_size=input_voxel_grid.voxel_size,
                    origin=input_voxel_grid.origin.copy()
                )
                timing_info.voxelization_time = 0.0
                logging.info(f"  Using cloned pre-computed voxel grid: {voxel_grid.shape}")
            else:
                if mesh is None:
                    raise ValueError("No mesh or voxel grid provided for simulation.")
                # Voxelize from mesh
                vox_start = time.perf_counter()
                voxelizer = Voxelizer(
                    voxel_size=self.voxel_size,
                    fill_interior=self.fill_interior,
                    max_memory_gb=self.memory_limit_gb
                )
                voxel_grid = voxelizer.voxelize(mesh)
                timing_info.voxelization_time = time.perf_counter() - vox_start
                logging.info(f"  Voxelized: {voxel_grid.shape} in {timing_info.voxelization_time:.2f}s")
            
            tracker.end_phase()  # End voxelization phase

            # ===== PHASE 1: Deferred Structure Generation =====
            initial_annotations = None  # Will be set if structure generation runs
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
                
                initial_annotations = None
                if method_name == "generate_lattice":
                    voxel_grid, initial_annotations = modifier.generate_lattice(config, progress_callback=struct_progress)
                elif method_name == "generate_random_voids":
                    voxel_grid, initial_annotations = modifier.generate_random_voids(config, progress_callback=struct_progress)
                else:
                    voxel_grid = modifier.grid
                    
                timing_info.structure_time = time.perf_counter() - struct_start
                logging.info(f"  Structure generation completed in {timing_info.structure_time:.2f}s")
                if initial_annotations:
                    logging.info(f"  Annotations: {len(initial_annotations.voids)} voids annotated")
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

            sim_input = ScientificData(
                primary_data=voxel_grid,
                secondary_data={},
                spatial_info={
                    "voxel_size_mm": voxel_grid.voxel_size,
                    "origin": voxel_grid.origin.copy(),
                },
                metadata={"stage": "voxelized"},
            )
            if isinstance(self.mesh_data, ScientificData):
                sim_input.metadata.update(
                    {k: v for k, v in self.mesh_data.metadata.items() if k not in sim_input.metadata}
                )
            
            if self.physics_mode:
                # Use physical simulation with polychromatic physics
                from simulation.physics import (
                    PhysicalCTSimulator,
                    PhysicsConfig,
                    PhysicalProcessConfig,
                )
                
                physics_config = PhysicsConfig(
                    kvp=self.physics_kvp,
                    tube_current_ma=self.physics_tube_current,
                    filtration_mm_al=self.physics_filtration,
                    exposure_multiplier=self.physics_exposure_multiplier,
                    energy_bins=self.physics_energy_bins,
                    use_gpu=self.use_gpu
                )
                
                simulator = PhysicalCTSimulator(
                    config=physics_config,
                    num_projections=self.num_projections
                )
                
                ct_result = simulator.process(
                    sim_input,
                    PhysicalProcessConfig(
                        material=self.material,
                        progress_callback=sim_progress,
                    ),
                )
                
            else:
                # Normal mode or fast mode: use SimpleCTSimulator
                # (Fast mode skips reconstruction in SimpleCTSimulator if applicable)
                simulator = SimpleCTSimulator(
                    num_projections=self.num_projections,
                    use_gpu=self.use_gpu
                )
                ct_result = simulator.process(
                    sim_input,
                    SimpleProcessConfig(
                        material=self.material,
                        progress_callback=sim_progress,
                    ),
                )

            ct_volume = ct_result.primary_data
            if not isinstance(ct_volume, CTVolume):
                raise TypeError(
                    "Simulator.process must return ScientificData.primary_data as CTVolume."
                )
            
            timing_info.simulation_time = time.perf_counter() - sim_start
            tracker.end_phase()  # End CT simulation phase
            
            # Get GPU-specific timing if available
            if hasattr(simulator, '_last_gpu_timing'):
                raw_gpu_timing = getattr(simulator, "_last_gpu_timing")
                if isinstance(raw_gpu_timing, dict):
                    try:
                        timing_info.gpu_timing = GPUSimulationTiming.from_mapping(raw_gpu_timing)
                    except (TypeError, ValueError):
                        logging.warning("Failed to parse GPU timing payload: %s", raw_gpu_timing)
            
            # ===== PHASE 3-4: Compression Simulation (only if enabled) =====
            compression_results = []
            if has_compression:
                logging.info("=" * 50)
                logging.info("PHASE 4: Compression Simulation workflow")
                tracker.start_phase(3)  # Compression Physics phase
                compression_results, compression_elapsed = run_compression_workflow(
                    voxel_grid=voxel_grid,
                    ct_volume=ct_volume,
                    simulator=simulator,
                    physics_mode=self.physics_mode,
                    material=self.material,
                    use_gpu=self.use_gpu,
                    compression_config=self.compression_config,
                    tracker=tracker,
                    initial_annotations=initial_annotations if self.structure_config else None,
                )

                timing_info.compression_time = compression_elapsed
                logging.info(f"  Compression + CT: {len(compression_results)} steps in {timing_info.compression_time:.2f}s")
            
            timing_info.total_time = time.perf_counter() - total_start
            
            # ===== Summary =====
            logging.info("=" * 50)
            logging.info("TIMING SUMMARY:")
            logging.info(f"  Voxelization:     {timing_info.voxelization_time:.2f}s")
            logging.info(f"  Structure Gen:    {timing_info.structure_time:.2f}s")
            logging.info(f"  CT Simulation:    {timing_info.simulation_time:.2f}s")
            if timing_info.compression_time is not None:
                logging.info(f"  Compression:      {timing_info.compression_time:.2f}s")
            logging.info(f"  TOTAL:            {timing_info.total_time:.2f}s")
            if timing_info.gpu_timing is not None:
                logging.info(f"  GPU Details: {timing_info.gpu_timing}")
            logging.info("=" * 50)
            
            self.progress.emit(1.0)
            # Emit annotations: use initial_annotations if no compression, otherwise embedded in compression_results
            annotations_out = None
            if self.structure_config is not None:
                annotations_out = initial_annotations
            self.finished.emit(ct_result, timing_info, compression_results, annotations_out)
            
        except (ValueError, TypeError, OSError, ImportError, RuntimeError) as e:
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
        window_width: float,
        initial_annotations=None,  # AnnotationSet for single-volume export
    ):
        super().__init__()
        self.volume_or_list = volume_or_list
        self.output_dir = output_dir
        self.window_center = window_center
        self.window_width = window_width
        self.initial_annotations = initial_annotations
    
    def run(self):
        try:
            all_files = export_volume_or_series(
                volume_or_list=self.volume_or_list,
                output_dir=self.output_dir,
                window_center=self.window_center,
                window_width=self.window_width,
                progress_callback=self.progress.emit,
                initial_annotations=self.initial_annotations,
            )
            self.finished.emit(all_files)
            
        except (ValueError, OSError, ImportError, RuntimeError) as e:
            import traceback
            logging.error(f"Export error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class StructureWorker(QThread):
    """Background worker for structure generation."""
    
    progress = Signal(float)
    finished = Signal(object, object)  # Emits (VoxelGrid, AnnotationSet)
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
            annotations = None
            if self.method_name == "generate_lattice":
                _, annotations = self.modifier.generate_lattice(self.config, progress_callback=on_progress)
            elif self.method_name == "generate_random_voids":
                _, annotations = self.modifier.generate_random_voids(self.config, progress_callback=on_progress)
            
            self.progress.emit(1.0)
            self.finished.emit(self.modifier.grid, annotations)
            
        except (ValueError, OSError, ImportError, RuntimeError) as e:
            import traceback
            logging.error(f"Structure generation error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))
