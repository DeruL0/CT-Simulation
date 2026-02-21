# Annotated 4DCT Simulation Output Report

## 1. Overview

This project (CT Simulation Software) is a high-performance Python-based CT simulation engine. Starting from a 3D STL model, it performs voxelization, microstructure generation, compression mechanics simulation, and CT forward/back projection reconstruction, then outputs an **annotated 4DCT DICOM time-series dataset**.

The dataset is designed for AI defect detection training and validation.

---

## 2. 4DCT Data Generation Pipeline

The complete data generation process is:

```
STL model → Voxelization → Microstructure/Defect Injection → Multi-step Compression Simulation → Per-step CT Forward Projection + Reconstruction → DICOM Export + Annotation Files
```

### 2.1 Voxelization

| Parameter | Default | Description |
|------|--------|------|
| `voxel_size_mm` | 0.5 mm | Voxel edge length |
| `fill_interior` | `True` | Fill interior of closed meshes |

- Input: triangle mesh (`.stl`)
- Output: `VoxelGrid` — 3D voxel grid `(Z, Y, X)`

### 2.2 Microstructure and Defect Injection

Handled by `StructureModifier`, supporting two structure types:

#### TPMS Lattice

| Parameter | Description |
|------|------|
| `lattice_type` | Gyroid / Schwarz Primitive / Diamond / Lidinoid / Split-P |
| `density_percent` | Target solid volume fraction (0–100%) |
| `cell_size_mm` | Unit lattice size (mm) |
| `preserve_shell` | Preserve outer shell |
| `shell_thickness_mm` | Shell thickness (mm) |

#### Random Defects (Voids)

| Parameter | Description |
|------|------|
| `shape` | Sphere / Cylinder / Ellipsoid |
| `density_percent` | Target porosity (0–100%) |
| `size_mean_mm` | Mean defect size (mm) |
| `size_std_mm` | Size standard deviation |
| `seed` | Random seed (optional, reproducible) |
| `preserve_shell` | Preserve outer shell |

**Key output**: each injected defect is recorded as a `VoidAnnotation`, forming the initial `AnnotationSet`.

### 2.3 Compression Mechanics Simulation

Orchestrated by `CompressionManager`, with physical and geometric modes:

| Parameter | Default | Description |
|------|--------|------|
| `total_compression` | 0.2 (20%) | Final total compression ratio |
| `num_steps` | 5 | Number of time steps (N+1 including initial state) |
| `poisson_ratio` | 0.3 | Material Poisson ratio |
| `axis` | Z | Compression axis (X / Y / Z) |
| `use_physics` | `True` | Use FEM elasticity solver |
| `downsample_factor` | 4 | Mechanics grid downsample factor |
| `solver_iterations` | 300 | Conjugate-gradient iterations |

- **Physical mode**: linear elasticity FEM computes displacement fields `(u_z, u_y, u_x)`, then warps the voxel grid.
- **Geometric mode**: affine scaling (`scipy.ndimage.zoom`), faster but less physically realistic.

For each time step, the system outputs a deformed voxel grid and **tracks/transforms all defect annotations**:
- Transform defect centers by cumulative scale factors
- Update defect radius/volume according to deformation scale
- Recompute voxel-space bounding boxes

### 2.4 CT Simulation and Reconstruction

Each deformed voxel grid is simulated independently:

| Parameter | Default | Description |
|------|--------|------|
| `num_projections` | 360 | Number of projection angles (0°–360°) |
| `kvp` | 120 kVp | Tube voltage |
| `tube_current_ma` | 200 mA | Tube current |
| `filtration_mm_al` | 2.5 mm Al | Aluminum filtration thickness |
| `energy_bins` | 10 | Number of spectral bins |
| `add_noise` | `True` | Poisson noise simulation |

- **Physical (polychromatic) mode**: multi-energy spectrum, energy-dependent attenuation, beam hardening
- **Simplified mode**: HU-based monoenergetic Radon + FBP reconstruction
- **Dual backend**: GPU (CuPy CUDA) / CPU (scikit-image)

Output: `CTVolume` — HU volume `(Z, Y, X)`, typically `float32`.

---

## 3. Output Directory Structure

The exported 4DCT time series is organized as:

```
<output_dir>/
├── Step_00/                          # Initial state (0% compression)
│   ├── CT_0000.dcm                   # Slice 1
│   ├── CT_0001.dcm                   # Slice 2
│   ├── ...
│   ├── CT_NNNN.dcm                   # Slice N
│   ├── annotations.json              # Full annotations (custom JSON)
│   ├── coco_annotations.json         # COCO detection format
│   └── labels.npy                    # Instance segmentation labels (int16)
├── Step_01/                          # Time step 1
│   ├── CT_*.dcm
│   ├── annotations.json
│   ├── coco_annotations.json
│   └── labels.npy
├── Step_02/
│   └── ...
├── ...
├── Step_NN/                          # Final compressed state
│   └── ...
└── annotations_summary.json          # Global time-series summary
```

---

## 4. DICOM Specification

Each DICOM file (`.dcm`) follows DICOM 3.0, SOP Class **CT Image Storage**.

### 4.1 Key DICOM Tags

| Tag | Name | Value / Description |
|-----|------|-----------|
| `(0008,0060)` | Modality | `CT` |
| `(0008,0008)` | Image Type | `DERIVED\\PRIMARY\\AXIAL` |
| `(0008,0070)` | Manufacturer | `CT Simulation Software` |
| `(0008,1090)` | Model Name | `CT Simulation v1.0` |
| `(0010,0010)` | Patient Name | Configurable (default `Anonymous^Patient`) |
| `(0010,0020)` | Patient ID | Configurable (default `SIMULATION001`) |
| `(0018,0050)` | Slice Thickness | Equals `voxel_size` (mm) |
| `(0018,0060)` | KVP | `120` |
| `(0020,0013)` | Instance Number | 1 ~ N |
| `(0020,0032)` | Image Position | `[origin_x, origin_y, origin_z + i × voxel_size]` |
| `(0020,0037)` | Image Orientation | `[1,0,0,0,1,0]` (axial) |
| `(0020,1041)` | Slice Location | Z world coordinate (mm) |
| `(0028,0010)` | Rows | Y dimension |
| `(0028,0011)` | Columns | X dimension |
| `(0028,0030)` | Pixel Spacing | `[voxel_size, voxel_size]` (mm) |
| `(0028,0100)` | Bits Allocated | `16` |
| `(0028,0101)` | Bits Stored | `16` |
| `(0028,0103)` | Pixel Representation | `0` (unsigned) |
| `(0028,1050)` | Window Center | default `40` HU |
| `(0028,1051)` | Window Width | default `400` HU |
| `(0028,1052)` | Rescale Intercept | `-1024` |
| `(0028,1053)` | Rescale Slope | `1.0` |
| `(0028,1054)` | Rescale Type | `HU` |

### 4.2 Pixel Encoding

- Transfer Syntax: **Explicit VR Little Endian**
- Photometric Interpretation: **MONOCHROME2**
- Storage formula: `Stored = (HU - (-1024)) / 1.0 = HU + 1024`
- HU recovery: `HU = Stored × RescaleSlope + RescaleIntercept`
- Pixel range: `[0, 65535]` (`uint16`)

### 4.3 UID Strategy

| UID Type | Generation |
|----------|----------|
| Study Instance UID | Unique per export |
| Series Instance UID | Unique per time step/series |
| SOP Instance UID | Unique per slice |
| Frame of Reference UID | Unique per export |

---

## 5. Annotation File Formats

### 5.1 `annotations.json` (custom format)

Contains full per-step 3D metadata:

```json
{
  "step_index": 0,
  "compression_ratio": 0.0,
  "voxel_size": 0.5,
  "volume_shape": [256, 256, 256],
  "origin": [0.0, 0.0, 0.0],
  "num_voids": 42,
  "voids": [
    {
      "id": 1,
      "shape": "sphere",
      "center_mm": [12.5, 30.2, 45.8],
      "radius_mm": 1.8,
      "volume_mm3": 24.43,
      "bbox_voxel_min": [80, 55, 85],
      "bbox_voxel_max": [88, 65, 93],
      "center_voxel": [84.0, 60.0, 89.0]
    }
  ]
}
```

**Field definitions:**

| Field | Type | Description |
|------|------|------|
| `id` | int | Unique defect ID |
| `shape` | string | `sphere` / `cylinder` / `ellipsoid` / `lattice_void` |
| `center_mm` | float[3] | World-space center (x, y, z), mm |
| `radius_mm` | float | Primary radius, mm |
| `volume_mm3` | float | Defect volume, mm³ |
| `radii_mm` | float[3] | Ellipsoid semi-axes (optional) |
| `axis_direction` | float[3] | Cylinder unit axis (optional) |
| `length_mm` | float | Cylinder length (optional) |
| `bbox_voxel_min` | int[3] | Voxel bbox min `[z, y, x]` |
| `bbox_voxel_max` | int[3] | Voxel bbox max `[z, y, x]` |
| `center_voxel` | float[3] | Derived voxel center |

### 5.2 `coco_annotations.json` (COCO format)

Generates axial-slice 2D bounding boxes compatible with COCO tools:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "slice_0000.png",
      "width": 256,
      "height": 256
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 5,
      "category_id": 1,
      "bbox": [85, 55, 8, 10],
      "area": 80,
      "iscrowd": 0,
      "void_id": 1,
      "void_shape": "sphere"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "void",
      "supercategory": "defect"
    }
  ]
}
```

- `bbox` format: `[x_min, y_min, width, height]`
- A defect gets one annotation per intersecting axial slice
- Extended keys `void_id` and `void_shape` support 3D-to-2D association

### 5.3 `labels.npy` (instance label volume)

- Format: NumPy `.npy`
- Data type: `int16`
- Shape: same as CT volume `(Z, Y, X)`
- Value meaning:
  - `0` = background
  - `> 0` = defect instance ID (matches `annotations.json` `id`)

Voxel assignment is computed by geometric tests (sphere distance, normalized ellipsoid distance, cylinder axial projection).

### 5.4 `annotations_summary.json` (time-series summary)

```json
{
  "num_steps": 6,
  "config": {
    "total_compression": 0.2,
    "num_steps": 5,
    "axis": "Z"
  },
  "steps": [
    {"step_index": 0, "compression_ratio": 0.0, "num_voids": 42},
    {"step_index": 1, "compression_ratio": 0.04, "num_voids": 42},
    {"step_index": 2, "compression_ratio": 0.08, "num_voids": 42},
    {"step_index": 3, "compression_ratio": 0.12, "num_voids": 42},
    {"step_index": 4, "compression_ratio": 0.16, "num_voids": 42},
    {"step_index": 5, "compression_ratio": 0.20, "num_voids": 42}
  ]
}
```

---

## 6. 4D Temporal Annotation Tracking

A key system feature is **consistent defect tracking throughout compression**.

### 6.1 Transformation workflow

```
Initial defect set (Step 0)
    │
    ▼ For each step (Step 1..N)
    ├── Compute cumulative scale: cumulative_scale = (s_z, s_y, s_x)
    │     s_z  = ∏(1 - Δε_i)          # compression-axis shrinkage
    │     s_xy = ∏(1 + ν·Δε_i)        # lateral expansion (Poisson effect)
    │
    ├── Transform center
    │     new_center = vol_center + (center - vol_center) × scale
    │
    ├── Transform geometry
    │     new_radius = radius × mean(|scale|)
    │     new_radii  = radii × |scale|           # ellipsoid
    │     new_length = length × mean(|scale|)    # cylinder
    │     new_volume = recomputed by shape formula
    │
    └── Recompute voxel bbox
          bbox_min/max = clamp(floor/ceil(center_voxel ± extent), 0, shape-1)
```

### 6.2 Multi-axis compression support

| Compression Axis | Internal handling | World-scale mapping |
|--------|----------|---------------------|
| Z (default) | No rotation, compress axis-0 | `scale_world = (s_z, s_y, s_x)` |
| X | `moveaxis(2→0)` before compression, restore by `moveaxis(0→2)` | `scale_world = (s_y, s_x, s_z)` |
| Y | `moveaxis(1→0)` before compression, restore by `moveaxis(0→1)` | `scale_world = (s_y, s_z, s_x)` |

---

## 7. Supported Material Database

HU values in exported DICOM volumes depend on selected material. Built-in materials include:

| Material | HU | Density (g/cm³) |
|------|--------|-------------|
| Air | −1000 | 0.001 |
| Water | 0 | 1.0 |
| Fat | −100 | 0.92 |
| Muscle | 40 | 1.06 |
| Bone (Cancellous) | 400 | 1.18 |
| Bone (Cortical) | 1000 | 1.85 |
| Titanium | 3000 | 4.51 |
| Steel | 3500 | 7.87 |
| Aluminum | 1500 | 2.70 |
| Plastic (PVC) | 100 | 1.40 |
| Plastic (PE) | −50 | 0.94 |
| Foam / Sponge | −800 | 0.05 |
| Rubber | −100 | 1.20 |
| Bread | −500 | 0.30 |
| Chocolate | 200 | 1.30 |
| ... | ... | ... |

---

## 8. Typical Data Usage

### 8.1 AI defect learning tasks

| Task | Annotation source | Dimension |
|----------|-------------|----------|
| 2D object detection | `coco_annotations.json` | Per-slice `(H, W)` |
| 3D instance segmentation | `labels.npy` | Full volume `(Z, Y, X)` |
| 3D semantic segmentation | `labels.npy` (binarized) | Full volume `(Z, Y, X)` |
| Temporal defect tracking | Multi-step `annotations.json` | 4D `(T, Z, Y, X)` |
| Defect statistics | `annotations.json` + `annotations_summary.json` | Tabular |

### 8.2 Recommended loading example

```python
import json
import numpy as np
import pydicom
from pathlib import Path

# Load one step CT series
step_dir = Path("output/Step_02")
dcm_files = sorted(step_dir.glob("CT_*.dcm"))
slices = [pydicom.dcmread(str(f)) for f in dcm_files]

# Reconstruct 3D HU volume
volume = np.stack([
    s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept)
    for s in slices
])

# Load annotations
with open(step_dir / "annotations.json") as f:
    annotations = json.load(f)

# Load instance labels
labels = np.load(step_dir / "labels.npy")

print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
print(f"Labels shape: {labels.shape}, unique IDs: {np.unique(labels)}")
print(f"Number of voids: {annotations['num_voids']}")
```

---

## 9. Limitations and Notes

1. **DICOM compatibility**: files are generated using `pydicom`, conforming to CT Image Storage and readable in common viewers (e.g., 3D Slicer, RadiAnt, OsiriX).

2. **Annotation precision**: annotations are generated from exact simulation geometry. Reconstructed CT appearance may differ at boundaries due to noise, artifacts, and voxel resolution.

3. **Compression tracking approximation**: in geometric mode, annotation transforms are affine approximations. In physical mode, volume deformation is field-based, while annotation update still uses global scale factors (adequate for moderate deformation).

4. **Memory management**: large volumes (e.g., 512³ / 1024³) rely on dynamic batching and optional disk offloading (`offload_to_disk`) to prevent OOM.

5. **Voxel resolution effects**: coarse voxel size (e.g., > 1 mm) can reduce bbox precision for small defects.

---

## 10. Summary

The 4DCT export provides a complete, multi-format annotated simulation dataset:

- **DICOM series**: standards-compliant CT image data with metadata
- **JSON annotations**: precise 3D defect geometry and metadata
- **COCO annotations**: 2D slice-level detection labels
- **Label volume**: voxel-wise instance segmentation ground truth
- **Temporal consistency**: annotation tracking across compression time steps

This format is designed for practical AI workflows, balancing medical imaging compliance and deep-learning ecosystem compatibility.
