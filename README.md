
# CT Simulation Software

A comprehensive Python-based desktop application for simulating Computed Tomography (CT) scans from 3D mesh models. This tool is designed for scientific and industrial applications, offering realistic physical X-ray simulation, internal structure generation (such as defects and lattices), and mechanical compression workflows.

## Overview

The CT Simulation Software allows users to import 3D models (STL, OBJ, PLY, etc.), convert them into high-resolution voxel grids, and simulate the CT scanning process using both simple and physical polychromatic X-ray models. It includes GPU acceleration via CuPy, making it suitable for generating large datasets for AI training or non-destructive testing (NDT) simulations.

## Key Features

-   **Mesh Import and Voxelization**
    
    -   Support for STL, PLY, OBJ, OFF, GLB, and GLTF formats.
        
    -   Configurable voxelization strategies (manual pitch, octree depth, target resolution).
        
    -   Automatic gap-filling for watertight meshes.
        
-   **Physical X-ray Simulation**
    
    -   Polychromatic X-ray spectrum generation (Kramers' law with tungsten characteristic peaks).
        
    -   Energy-dependent attenuation using built-in NIST XCOM material data (metals, plastics, tissues).
        
    -   Simulation of physical artifacts: beam hardening, Poisson photon noise, X-ray scatter, and motion blur.
        
-   **Industrial Structure and Defect Generation**
    
    -   Automated TPMS lattice generation (Gyroid, Schwarz, Lidinoid) with density control.
        
    -   Random defect/void generation (spheres, cylinders, ellipsoids) to simulate porosity.
        
    -   Manual geometric modifiers.
        
    -   Export of AI-ready defect annotations (COCO format, instance segmentation label volumes).
        
-   **Mechanical Compression Simulation**
    
    -   GPU-accelerated linear elasticity solver using Finite Difference Methods.
        
    -   Simulates physical deformation under uniaxial compression.
        
    -   Generates multi-step time-series CT simulations of the compression process.
        
-   **High-Performance Computing**
    
    -   Custom fused CUDA kernels (via CuPy) for lightning-fast Radon and inverse Radon (FBP) transforms.
        
    -   Dynamic GPU memory management and batch processing to prevent out-of-memory errors on large volumes.
        
    -   Fallback CPU implementation utilizing scikit-image.
        
-   **Visualization and Export**
    
    -   Interactive 3D mesh and isosurface viewer (powered by PyVista).
        
    -   2D slice viewer with medical window/level presets (powered by PyQtGraph).
        
    -   DICOM series export with comprehensive, customizable metadata.
        

## Prerequisites

-   Python 3.9 or higher
    
-   A compatible NVIDIA GPU (highly recommended for performance and physical simulation modes)
    

### Core Dependencies

-   `PySide6` (GUI framework)
    
-   `numpy`
    
-   `scipy`
    
-   `trimesh` (Mesh processing)
    
-   `scikit-image` (CPU-based Radon transforms and image processing)
    
-   `pydicom` (DICOM export)
    

### Optional but Recommended Dependencies

-   `cupy` (Required for GPU acceleration and the mechanical elasticity solver)
    
-   `pyvista` and `pyvistaqt` (Required for the 3D Viewer tab)
    
-   `pyqtgraph` (Required for high-performance 2D slice viewing)
    

## Installation

1.  Clone the repository:
    
    ```
    git clone [https://github.com/yourusername/ct-simulation-software.git](https://github.com/yourusername/ct-simulation-software.git)
    cd ct-simulation-software
    ```
    
2.  Create a virtual environment (recommended):
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
    
3.  Install the required dependencies:
    
    ```
    pip install numpy scipy trimesh scikit-image pydicom PySide6
    ```
    
4.  Install the optional visualization libraries:
    
    ```
    pip install pyvista pyvistaqt pyqtgraph
    ```
    
5.  Install CuPy for your specific CUDA version (if using an NVIDIA GPU). For example, for CUDA 11.x:
    
    ```
    pip install cupy-cuda11x
    ```
    

## Usage

Start the application by running the main entry point:

```
python main.py
```

### Basic Workflow

1.  **Import Model:** Open the "Loader Panel" and select a 3D mesh file. You can scale the model and select a base material (e.g., Titanium, Cortical Bone) from the database.
    
2.  **Configure Parameters:** Set your target voxel size, CT simulation parameters (kVp, mA, Projections), and toggle GPU acceleration in the "Parameters Panel".
    
3.  **Add Internal Structures (Optional):** Use the "Structure Generation" panel to add TPMS lattices or random porosity defects to your solid model.
    
4.  **Run Simulation:** Click "Run Simulation". The software will track progress across voxelization, structure generation, and CT simulation phases.
    
5.  **Visualize:** Use the "2D Slices" tab to inspect the cross-sections and adjust Window/Level settings. Use the "3D View" tab to see the reconstructed isosurface.
    
6.  **Export:** Click "Export DICOM" to save the simulated volume as a standard DICOM image series for use in third-party medical or industrial inspection software.
    

## Project Structure

-   `main.py`: Application entry point.
    
-   `config.py`: Default configuration and constants.
    
-   `core/`: Base data structures (`ScientificData`) and the centralized `DataManager`.
    
-   `exporters/`: Contains the `DICOMExporter` for saving simulation results.
    
-   `gui/`: PySide6 user interface components, panels, styles, and background worker threads.
    
-   `loaders/`: `MeshLoader` utilizing trimesh for handling various 3D file formats.
    
-   `simulation/`: The core algorithmic engine.
    
    -   `backends/`: CPU and GPU implementations of the Radon/IRadon transforms.
        
    -   `mechanics/`: Linear elasticity solver for compression simulation.
        
    -   `physics/`: Polychromatic X-ray spectrum generation, material attenuation databases, and scatter modeling.
        
    -   `structures/`: TPMS lattice generation, random defect placement, and AI annotation exporters.
        
-   `visualization/`: 2D slice and 3D mesh rendering wrappers.
    

## License

[Specify your license here, e.g., MIT License, GPL-3.0, etc.]