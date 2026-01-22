# CT Simulation Software

A high-performance scientific application for simulating Computed Tomography (CT) scans from 3D STL models.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **3D Model Import**: Load distinct STL files for simulation.
- **Voxelization**: Convert vector meshes into voxel grids with configurable resolution.
- **Microstructure Generation**:
  - **TPMS Lattices**: Generate Gyroid, Schwarz, and other lattices with density control.
  - **Defect Generation**: Inject random voids (Spheres, Cylinders, Ellipsoids) with statistical size distributions.
  - **Shell Preservation**: Protect outer layers of the object from modification.
- **Physical CT Simulation**:
  - **Physics Model**: Polychromatic X-ray sources (spectrum generation), energy-dependent attenuation, and beam hardening effects.
  - **Dual Backend**:
    - **GPU**: High-performance CUDA implementation using `CuPy` (up to 50x faster) + Dynamic Batching.
    - **CPU**: Reliable fallback using `scikit-image`.
  - Configurable physics: kVp range, filtration, photon noise.
- **Compression Simulation**:
  - **Physics-Based**: Linear elasticity simulation using Finite Element Method (FEM).
  - **GPU Acceleration**: Fast conjugate gradient solver on GPU.
  - **Time-Series**: 4D visualization of compression steps with adjustable slider.
  - **Workflow Integration**: Seamlessly chain Voxelization -> Structure -> Compression -> CT Simulation.
- **Visualization**:
  - **3D Mesh Viewer**: Inspect input STL models with realistic lighting.
  - **2D Slice Viewer**: Interactive axial, coronal, and sagittal views.
  - **Direct Volume Rendering**: 3D visualization of reconstructed volumes.
- **Export**: Save simulated data as DICOM series with comprehensive metadata.

## Project Structure

```text
Simulation/
├── core/               # Data management and abstract base classes
├── exporters/          # Data export (DICOM)
├── gui/                # UI components
│   ├── main_window.py
│   ├── pages/          # Structure generation sub-pages
│   └── panels/         # Reusable config panels
├── loaders/            # File loaders (STL)
├── simulation/         # Core algorithms
│   ├── backends/       # GPU (CuPy) and CPU (skimage) backends
│   ├── mechanics/      # NEW: Compression and Elasticity Solver
│   ├── physics/        # X-ray physics (Spectrum, Attenuation)
│   ├── structures/     # Lattice and Defect generation
│   ├── simple_simulator.py # Fast preview simulator
│   └── voxelizer.py    # Voxelization logic
└── visualization/      # Rendering components (2D/3D viewers)
```

## Installation

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install numpy scipy pyside6 pyvista pyvistaqt trimesh scikit-image
   ```

3. (Optional) For GPU acceleration (Highly Recommended):

   ```bash
   pip install cupy-cuda12x  # Match your CUDA version (e.g., cupy-cuda11x, cupy-cuda12x)
   ```

## Usage

Run the main application:

```bash
python main.py
```

### Workflow

1. **Load STL**: Use the "Import STL" panel to load a 3D model.
2. **Structure (Optional)**:
   - Go to "Structure Generation" to apply internal patterns (Lattices) or Defects.
   - Click "Generate" (applied automatically during simulation pipeline).
3. **Compression (Optional)**:
   - Configure compression axis (X/Y/Z), ratio, and number of time steps.
   - Enables 4D time-series visualization.
4. **Configure Simulation**:
   - Set Material (human tissues or industrial materials).
   - Configure Scanner Physics (kVp, Filtration).
5. **Simulate**: Click "Run Simulation".
   - The app automatically detects GPU availability.
   - Real-time progress is shown for Voxelization -> Modification -> Forward Projection -> Reconstruction.
6. **Visualize & Export**: Inspect slices and export to DICOM.

## Performance

- **GPU Acceleration**: Uses custom CUDA kernels (via CuPy) for Radon and Inverse Radon transforms.
- **Memory Management**: Implements dynamic batching to handle large volumes (e.g., 512^3 or 1024^3) without OOM errors on standard GPUs.
