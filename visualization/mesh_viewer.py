"""
Mesh Viewer

Provides 3D visualization for STL meshes and CT volumes using PyVista.
"""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QComboBox, QLabel, QCheckBox
)
from PySide6.QtCore import Qt

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import trimesh
except ImportError:
    trimesh = None


class MeshViewer(QWidget):
    """3D viewer for STL meshes and CT volumes."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._mesh = None
        self._ct_volume = None
        self._plotter = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D View
        view_group = QGroupBox("3D View")
        view_layout = QVBoxLayout(view_group)
        
        if HAS_PYVISTA:
            # Create PyVista Qt interactor
            self._plotter = QtInteractor(self)
            self._plotter.set_background('#F5F5F5')  # Light gray background
            self._setup_lighting()
            self._plotter.add_axes()
            view_layout.addWidget(self._plotter.interactor)
        else:
            placeholder = QLabel("3D Viewer requires pyvistaqt.\nInstall with: pip install pyvistaqt")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #666; padding: 20px;")
            view_layout.addWidget(placeholder)
        
        layout.addWidget(view_group, stretch=1)
        
        # View controls
        controls_group = QGroupBox("View Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # View preset dropdown
        controls_layout.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Isometric", "Front", "Back", "Left", "Right", "Top", "Bottom"])
        self._view_combo.currentTextChanged.connect(self._on_view_changed)
        controls_layout.addWidget(self._view_combo)
        
        controls_layout.addStretch()
        
        # Show edges checkbox
        self._edges_check = QCheckBox("Show Edges")
        self._edges_check.setChecked(False)
        self._edges_check.stateChanged.connect(self._on_edges_toggled)
        controls_layout.addWidget(self._edges_check)
        
        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._on_reset_view)
        controls_layout.addWidget(reset_btn)
        
        layout.addWidget(controls_group)
    
    def _setup_lighting(self) -> None:
        """Set up 3-point lighting for realistic rendering."""
        if self._plotter is None:
            return
        
        # Remove default lighting
        self._plotter.remove_all_lights()
        
        # Key light (main light source - bright, from upper right)
        key_light = pv.Light(
            position=(5, 5, 10),
            focal_point=(0, 0, 0),
            color='white',
            intensity=1.0
        )
        key_light.positional = False  # Directional light
        self._plotter.add_light(key_light)
        
        # Fill light (softer, from left side to reduce shadows)
        fill_light = pv.Light(
            position=(-5, 0, 5),
            focal_point=(0, 0, 0),
            color='#E8E8FF',  # Slight blue tint
            intensity=0.5
        )
        fill_light.positional = False
        self._plotter.add_light(fill_light)
        
        # Back light (rim light from behind)
        back_light = pv.Light(
            position=(0, -5, -5),
            focal_point=(0, 0, 0),
            color='#FFFAF0',  # Warm white
            intensity=0.3
        )
        back_light.positional = False
        self._plotter.add_light(back_light)
        
        # Ambient light for overall illumination
        ambient_light = pv.Light(
            light_type='headlight',
            intensity=0.2
        )
        self._plotter.add_light(ambient_light)
    
    def set_mesh(self, mesh: "trimesh.Trimesh") -> None:
        """
        Display a trimesh mesh in the 3D viewer.
        
        Args:
            mesh: trimesh.Trimesh object to display
        """
        if not HAS_PYVISTA or self._plotter is None:
            return
        
        self._mesh = mesh
        
        # Clear previous actors
        self._plotter.clear()
        self._setup_lighting()  # Re-apply lighting after clear
        
        # Convert trimesh to PyVista
        faces = mesh.faces
        # PyVista requires face array with count prefix
        pv_faces = np.column_stack([
            np.full(len(faces), 3),
            faces
        ]).ravel()
        
        pv_mesh = pv.PolyData(mesh.vertices, pv_faces)
        
        # Compute normals for better shading
        pv_mesh.compute_normals(inplace=True)
        
        # Add mesh to plotter with enhanced lighting
        self._plotter.add_mesh(
            pv_mesh,
            color='#D0D0D0',  # Light gray
            show_edges=self._edges_check.isChecked(),
            edge_color='#666666',
            lighting=True,
            smooth_shading=True,
            specular=0.5,       # Specular reflection
            specular_power=15,  # Specular highlight sharpness
            ambient=0.2,        # Ambient reflection
            diffuse=0.8         # Diffuse reflection
        )
        
        # Reset camera
        self._plotter.reset_camera()
        self._plotter.add_axes()
    
    def set_ct_volume(self, ct_data: np.ndarray, voxel_size: float, threshold: float = 0.0) -> None:
        """
        Display CT volume as 3D isosurface.
        
        Args:
            ct_data: 3D numpy array of HU values (Z, Y, X)
            voxel_size: Voxel edge length in mm
            threshold: HU threshold for isosurface extraction
        """
        if not HAS_PYVISTA or self._plotter is None:
            return
        
        self._ct_volume = ct_data
        
        # Clear previous actors
        self._plotter.clear()
        self._setup_lighting()  # Re-apply lighting after clear
        
        # Create uniform grid
        grid = pv.ImageData()
        grid.dimensions = np.array(ct_data.shape) + 1
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.cell_data["values"] = ct_data.flatten(order="F")
        
        # Extract isosurface
        try:
            contour = grid.contour([threshold])
            if contour.n_points > 0:
                # Compute normals for smooth shading
                contour.compute_normals(inplace=True)
                
                self._plotter.add_mesh(
                    contour,
                    color='#E0E0E0',
                    show_edges=False,
                    lighting=True,
                    smooth_shading=True,
                    specular=0.5,
                    specular_power=15,
                    ambient=0.2,
                    diffuse=0.8
                )
        except Exception:
            # Fallback: show volume rendering
            self._plotter.add_volume(
                grid,
                scalars="values",
                opacity="sigmoid",
                cmap="bone"
            )
        
        self._plotter.reset_camera()
        self._plotter.add_axes()
    
    def _on_view_changed(self, view_name: str) -> None:
        """Handle view preset change."""
        if self._plotter is None:
            return
        
        # Define view vectors as numeric arrays (x, y, z)
        views = {
            "Isometric": (1, 1, 1),      # Isometric view
            "Front": (0, -1, 0),         # Looking from front (negative Y)
            "Back": (0, 1, 0),           # Looking from back (positive Y)
            "Left": (-1, 0, 0),          # Looking from left (negative X)
            "Right": (1, 0, 0),          # Looking from right (positive X)
            "Top": (0, 0, 1),            # Looking from top (positive Z)
            "Bottom": (0, 0, -1)         # Looking from bottom (negative Z)
        }
        
        if view_name in views:
            self._plotter.view_vector(views[view_name])
            self._plotter.reset_camera()
    
    def _on_edges_toggled(self, state: int) -> None:
        """Handle edge visibility toggle."""
        if self._mesh is not None:
            self.set_mesh(self._mesh)
    
    def _on_reset_view(self) -> None:
        """Reset camera to default isometric view."""
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._view_combo.setCurrentText("Isometric")
    
    def clear(self) -> None:
        """Clear the 3D viewer."""
        if self._plotter is not None:
            self._plotter.clear()
            self._plotter.add_axes()
        self._mesh = None
        self._ct_volume = None
