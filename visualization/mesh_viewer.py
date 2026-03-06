"""
Mesh Viewer

Provides 3D visualization for STL meshes and CT volumes using PyVista.
"""

from typing import Optional
import logging
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QComboBox, QLabel, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal

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

from .volume_viewer import VolumeViewer


class MeshViewer(QWidget):
    """3D viewer for STL meshes and CT volumes."""
    threshold_changed = Signal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._viewer_state = VolumeViewer()
        self._plotter = None
        self._current_volume_data: Optional[np.ndarray] = None
        self._current_voxel_size: float = 1.0
        self._threshold_value: float = 0.0
        
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

        volume_controls_group = QGroupBox("Volume Controls")
        volume_controls_layout = QHBoxLayout(volume_controls_group)
        volume_controls_layout.addWidget(QLabel("Threshold:"))
        self._threshold_edit = QLineEdit("0.0")
        self._threshold_edit.setPlaceholderText("Enter threshold value")
        self._threshold_edit.setToolTip("Isosurface threshold value (no fixed numeric limits).")
        self._threshold_edit.editingFinished.connect(self._on_threshold_edited)
        self._threshold_edit.setEnabled(False)
        volume_controls_layout.addWidget(self._threshold_edit)
        layout.addWidget(volume_controls_group)

    @staticmethod
    def _format_threshold(value: float) -> str:
        return f"{value:.10g}"

    @staticmethod
    def _parse_threshold(value: str) -> Optional[float]:
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _set_threshold_value(self, value: float, *, emit_signal: bool) -> None:
        self._threshold_value = float(value)
        self._threshold_edit.setText(self._format_threshold(self._threshold_value))
        if emit_signal:
            self.threshold_changed.emit(self._threshold_value)

    def _render_current_volume(self) -> None:
        """Render currently loaded CT volume using the active threshold value."""
        if not HAS_PYVISTA or self._plotter is None or self._current_volume_data is None:
            return

        # CTVolume convention is (Z, Y, X), while PyVista ImageData expects (X, Y, Z).
        ct_data_zyx = self._current_volume_data
        ct_data_xyz = np.transpose(ct_data_zyx, (2, 1, 0))
        voxel_size = self._current_voxel_size
        threshold = self._threshold_value
        self._viewer_state.set_volume(ct_data_zyx, voxel_size=voxel_size, threshold=threshold)

        # Clear previous actors
        self._plotter.clear()
        self._setup_lighting()  # Re-apply lighting after clear

        # Create uniform grid
        grid = pv.ImageData()
        grid.dimensions = np.array(ct_data_xyz.shape) + 1
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.cell_data["values"] = ct_data_xyz.flatten(order="F")

        # Extract isosurface
        try:
            contour = grid.contour([threshold])
            if contour.n_points > 0:
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
            self._plotter.add_volume(
                grid,
                scalars="values",
                opacity="sigmoid",
                cmap="bone"
            )

        self._plotter.reset_camera()
        self._plotter.add_axes()
    
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
        
        self._viewer_state.set_mesh(mesh)
        self._current_volume_data = None
        self._threshold_edit.setEnabled(False)
        
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
            show_edges=self._viewer_state.show_edges,
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
    
    def set_ct_volume(
        self,
        ct_data: np.ndarray,
        voxel_size: float,
        threshold: float = 0.0,
        *,
        preserve_threshold: bool = False,
    ) -> None:
        """
        Display CT volume as 3D isosurface.
        
        Args:
            ct_data: 3D numpy array of linear attenuation values (cm^-1, Z, Y, X)
            voxel_size: Voxel edge length in mm
            threshold: Threshold for isosurface extraction
        """
        if not HAS_PYVISTA or self._plotter is None:
            return

        self._current_volume_data = ct_data
        self._current_voxel_size = float(voxel_size)
        self._threshold_edit.setEnabled(True)

        if not preserve_threshold:
            self._set_threshold_value(float(threshold), emit_signal=True)

        self._render_current_volume()
    
    def _on_view_changed(self, view_name: str) -> None:
        """Handle view preset change."""
        if self._plotter is None:
            return

        preset = self._viewer_state.set_view(view_name)
        vector = preset.get("vector", (1.0, 1.0, 1.0))
        self._plotter.view_vector(vector)
        self._plotter.reset_camera()
    
    def _on_edges_toggled(self, state: int) -> None:
        """Handle edge visibility toggle."""
        self._viewer_state.set_edges_visible(state != 0)
        if self._viewer_state.mesh is not None:
            self.set_mesh(self._viewer_state.mesh)
    
    def _on_reset_view(self) -> None:
        """Reset camera to default isometric view."""
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._view_combo.setCurrentText("Isometric")

    def _on_threshold_edited(self) -> None:
        """Apply user threshold input without imposing numeric limits."""
        parsed = self._parse_threshold(self._threshold_edit.text())
        if parsed is None:
            self._threshold_edit.setText(self._format_threshold(self._threshold_value))
            return

        self._set_threshold_value(parsed, emit_signal=True)
        if self._current_volume_data is not None:
            self._render_current_volume()
    
    def clear(self) -> None:
        """Clear the 3D viewer."""
        if self._plotter is not None:
            self._plotter.clear()
            self._plotter.add_axes()
        self._current_volume_data = None
        self._threshold_edit.setEnabled(False)
        self._viewer_state.clear()

    @property
    def threshold(self) -> float:
        """Get current threshold value."""
        return self._threshold_value

    def reset_display_state(self) -> None:
        """Reset clip/selection style controls to defaults."""
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentText("Isometric")
        self._view_combo.blockSignals(False)

        self._edges_check.blockSignals(True)
        self._edges_check.setChecked(False)
        self._edges_check.blockSignals(False)
        self._viewer_state.set_edges_visible(False)

        self._set_threshold_value(0.0, emit_signal=True)

    def cleanup(self) -> None:
        """Release PyVista/VTK resources before application shutdown."""
        plotter = self._plotter
        self._plotter = None
        if plotter is None:
            return

        try:
            plotter.clear()
        except Exception as exc:
            logging.debug("Plotter clear during cleanup failed: %s", exc)

        try:
            interactor = getattr(plotter, "interactor", None)
            if interactor is not None:
                interactor.close()
        except Exception as exc:
            logging.debug("Plotter interactor close failed: %s", exc)

        try:
            plotter.close()
        except Exception as exc:
            logging.debug("Plotter close failed: %s", exc)

    def closeEvent(self, event) -> None:
        """Ensure native rendering resources are torn down with the widget."""
        self.cleanup()
        super().closeEvent(event)
