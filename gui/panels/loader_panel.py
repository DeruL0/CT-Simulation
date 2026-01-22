"""
Mesh Import Panel

Provides UI controls for importing and viewing 3D mesh file information.
Supports STL, PLY, OBJ, OFF, GLB, and GLTF formats.
"""

from pathlib import Path
from typing import Optional, Callable
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QFormLayout, QMessageBox
)
from PySide6.QtCore import Signal

from loaders import MeshLoader, MeshInfo, SUPPORTED_EXTENSIONS
from simulation.materials import MaterialType
from ..utils import create_spinbox


class LoaderPanel(QWidget):
    """Panel for 3D mesh file import and material selection."""
    
    # Signal emitted when a new mesh file is loaded
    stl_loaded = Signal(object)  # Emits the MeshLoader object (keep signal name for compatibility)
    # Signal emitted when mesh is scaled
    mesh_scaled = Signal(float)  # Emits the scale factor
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._loader: Optional[MeshLoader] = None
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Import section
        import_group = QGroupBox("Import 3D Mesh")
        import_layout = QVBoxLayout(import_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self._file_label = QLabel("No file selected")
        self._file_label.setObjectName("secondaryLabel")
        self._file_label.setWordWrap(True)
        
        self._import_btn = QPushButton("📂 Import 3D Mesh File...")
        self._import_btn.setMinimumHeight(32)
        # White background, black text, bold
        self._import_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border: 1px solid #999;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """)
        self._import_btn.clicked.connect(self._on_browse_clicked)
        
        file_layout.addWidget(self._file_label, stretch=1)
        file_layout.addWidget(self._import_btn)
        import_layout.addLayout(file_layout)
        
        layout.addWidget(import_group)
        
        # Ensure panel doesn't get crushed
        self.setMinimumHeight(200)
        
        # Mesh info section
        info_group = QGroupBox("Mesh Information")
        info_layout = QFormLayout(info_group)
        
        self._vertices_label = QLabel("-")
        self._faces_label = QLabel("-")
        self._dimensions_label = QLabel("-")
        self._volume_label = QLabel("-")
        self._watertight_label = QLabel("-")
        
        info_layout.addRow("Vertices:", self._vertices_label)
        info_layout.addRow("Faces:", self._faces_label)
        info_layout.addRow("Dimensions:", self._dimensions_label)
        info_layout.addRow("Volume:", self._volume_label)
        info_layout.addRow("Watertight:", self._watertight_label)
        
        layout.addWidget(info_group)
        
        # Scale section
        scale_group = QGroupBox("Scale Model")
        scale_layout = QHBoxLayout(scale_group)
        
        self._scale_spin = create_spinbox(
            1.0, 0.01, 100.0, step=0.1, decimals=2, suffix="×"
        )
        self._scale_spin.setToolTip("Scale factor (1.0 = original size)")
        scale_layout.addWidget(self._scale_spin, stretch=1)
        
        self._apply_scale_btn = QPushButton("Apply")
        self._apply_scale_btn.setEnabled(False)
        self._apply_scale_btn.clicked.connect(self._on_apply_scale)
        scale_layout.addWidget(self._apply_scale_btn)
        
        layout.addWidget(scale_group)
        
        # Material selection
        material_group = QGroupBox("Material")
        material_layout = QVBoxLayout(material_group)
        
        self._material_combo = QComboBox()
        for mat_type in MaterialType:
            if mat_type != MaterialType.CUSTOM:
                # Convert enum name to display name
                display_name = mat_type.value.replace("_", " ").title()
                self._material_combo.addItem(display_name, mat_type)
        
        # Set default to cortical bone
        default_index = self._material_combo.findData(MaterialType.BONE_CORTICAL)
        if default_index >= 0:
            self._material_combo.setCurrentIndex(default_index)
        
        material_layout.addWidget(self._material_combo)
        layout.addWidget(material_group)
        
        # Stretch at bottom
        layout.addStretch()
    
    def _on_browse_clicked(self) -> None:
        """Handle browse button click."""
        # Build file filter from supported extensions
        ext_str = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        filter_str = f"3D Mesh Files ({ext_str});;STL Files (*.stl);;PLY Files (*.ply);;OBJ Files (*.obj);;All Files (*.*)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D Mesh File",
            "",
            filter_str
        )
        
        if file_path:
            self.load_stl(file_path)
    
    def load_stl(self, file_path: str) -> bool:
        """
        Load a 3D mesh file.
        
        Args:
            file_path: Path to the mesh file (STL, PLY, OBJ, etc.)
            
        Returns:
            True if loading was successful
        """
        try:
            self._loader = MeshLoader()
            self._loader.load(file_path)
            
            # Update file label
            path = Path(file_path)
            self._file_label.setText(path.name)
            
            # Update mesh info
            self._update_mesh_info()
            
            # Enable scale button
            self._apply_scale_btn.setEnabled(True)
            self._scale_spin.setValue(1.0)
            
            # Emit signal
            self.stl_loaded.emit(self._loader)
            
            return True
            
        except Exception as e:
            self._file_label.setText("Import Failed")
            self._clear_mesh_info()
            
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to load mesh file:\n\n{str(e)}"
            )
            return False
    
    def _update_mesh_info(self) -> None:
        """Update mesh information labels."""
        if self._loader is None or self._loader.info is None:
            self._clear_mesh_info()
            return
        
        info = self._loader.info
        
        self._vertices_label.setText(f"{info.num_vertices:,}")
        self._faces_label.setText(f"{info.num_faces:,}")
        
        dims = info.dimensions
        self._dimensions_label.setText(
            f"{dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm"
        )
        
        if info.volume is not None:
            self._volume_label.setText(f"{info.volume:.2f} mm³")
        else:
            self._volume_label.setText("N/A (not watertight)")
        
        self._watertight_label.setText("Yes" if info.is_watertight else "No")
    
    def _clear_mesh_info(self) -> None:
        """Clear mesh information labels."""
        self._vertices_label.setText("-")
        self._faces_label.setText("-")
        self._dimensions_label.setText("-")
        self._volume_label.setText("-")
        self._watertight_label.setText("-")
        self._apply_scale_btn.setEnabled(False)
    
    def _on_apply_scale(self) -> None:
        """Apply scale factor to the loaded mesh."""
        if self._loader is None or self._loader.mesh is None:
            return
        
        scale_factor = self._scale_spin.value()
        if scale_factor == 1.0:
            return  # No scaling needed
        
        try:
            # Scale the mesh
            self._loader.mesh.apply_scale(scale_factor)
            
            # Update info (recalculate after scaling)
            self._loader.info = None  # Clear cached info
            self._loader._compute_info()  # Recalculate
            self._update_mesh_info()
            
            # Reset scale spinbox
            self._scale_spin.setValue(1.0)
            
            # Emit signal
            self.mesh_scaled.emit(scale_factor)
            
            QMessageBox.information(
                self,
                "Scale Applied",
                f"Model scaled by {scale_factor:.2f}×"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Scale Error",
                f"Failed to scale mesh:\n\n{str(e)}"
            )
    
    @property
    def loader(self) -> Optional[MeshLoader]:
        """Get the current mesh loader."""
        return self._loader
    
    @property
    def selected_material(self) -> MaterialType:
        """Get the currently selected material type."""
        return self._material_combo.currentData()
