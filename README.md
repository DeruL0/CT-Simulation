# CT Simulation Software

A high-performance scientific application for simulating Computed Tomography (CT) scans from 3D STL models.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

## Features

- **3D Model Import**: Load distinct STL files for simulation.
- **Voxelization**: Convert vector meshes into voxel grids with configurable resolution.
- **CT Simulation**:
  - **Dual Backend**:
    - **CPU**: Threaded implementation using `scikit-image`
    - **GPU**: Fully vectorized accelerated implementation using `CuPy` (up to 50x faster)
  - Configurable parameters: Projections, noise level, material types.
- **Visualization**:
  - **3D Mesh Viewer**: Inspect input STL models with realistic lighting.
  - **2D Slice Viewer**: Interactive axial, coronal, and sagittal views of the simulated volume.
  - **Direct Volume Rendering**: 3D visualization of the reconstruction.
- **Export**: Save simulated data as DICOM series.

## Project Structure

```
Simulation/
├── core/               # Data management and abstract base classes
├── exporters/          # Data export (DICOM)
├── gui/                # UI components (MainWindow, Panels, Workers)
├── loaders/            # File loaders (STL)
├── simulation/         # Core algorithms (Voxelization, CT Physics)
│   └── backends/       # CPU and GPU simulation implementations
└── visualization/      # Rendering components (2D/3D viewers)
```

## Installation

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install numpy scipy pyside6 pyvista pyvistaqt trimesh scikit-image
   ```

3. (Optional) For GPU acceleration:

   ```bash
   pip install cupy-cuda12x  # Match your CUDA version
   ```

## Usage

Run the main application:

```bash
python main.py
```

### Workflow

1. **Load STL**: Use the "Import STL" panel to load a 3D model.
2. **Configure**: Set material properties (e.g., Bone, Soft Tissue) and simulation parameters.
3. **Simulate**: Click "Run Simulation". Choose CPU or GPU backend.
4. **Visualize**: Inspect the results in the 2D/3D viewers.
5. **Export**: Export the result to DICOM for use in other medical software.
