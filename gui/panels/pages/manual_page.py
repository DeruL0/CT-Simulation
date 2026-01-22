"""
Manual Modifiers Page

Page for manually adding defects to the structure.
"""

from typing import Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QPushButton, QSizePolicy
)

from PySide6.QtCore import Signal
from ...utils import create_spinbox


class ManualPage(QWidget):
    """
    UI for manually adding geometric voids.
    """
    
    # Signals for actions
    add_sphere = Signal(tuple, float)  # center (x,y,z), radius
    add_cylinder = Signal(tuple, tuple, float)  # start, end, radius
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Sphere void
        sphere_group = QGroupBox("Add Sphere")
        sphere_layout = QFormLayout(sphere_group)
        
        self._sphere_x = create_spinbox(
            0.0, -1000, 1000, suffix=" mm"
        )
        sphere_layout.addRow("Center X:", self._sphere_x)
        
        self._sphere_y = create_spinbox(
            0.0, -1000, 1000, suffix=" mm"
        )
        sphere_layout.addRow("Center Y:", self._sphere_y)
        
        self._sphere_z = create_spinbox(
            0.0, -1000, 1000, suffix=" mm"
        )
        sphere_layout.addRow("Center Z:", self._sphere_z)
        
        self._sphere_r = create_spinbox(
            5.0, 0.1, 100, suffix=" mm"
        )
        sphere_layout.addRow("Radius:", self._sphere_r)
        
        self._add_sphere_btn = QPushButton("Add Sphere")
        self._add_sphere_btn.clicked.connect(self._on_sphere_clicked)
        sphere_layout.addRow(self._add_sphere_btn)
        
        layout.addWidget(sphere_group)
        
        # Cylinder void
        cyl_group = QGroupBox("Add Cylinder")
        cyl_layout = QFormLayout(cyl_group)
        
        self._cyl_x1 = create_spinbox(
            0.0, -1000, 1000, suffix=" mm"
        )
        cyl_layout.addRow("Start X:", self._cyl_x1)
        
        self._cyl_y1 = create_spinbox(
            0.0, -1000, 1000, suffix=" mm"
        )
        cyl_layout.addRow("Start Y:", self._cyl_y1)
        
        self._cyl_z1 = create_spinbox(
            0.0, -1000, 1000, suffix=" mm"
        )
        cyl_layout.addRow("Start Z:", self._cyl_z1)
        
        self._cyl_x2 = create_spinbox(
            10.0, -1000, 1000, suffix=" mm"
        )
        cyl_layout.addRow("End X:", self._cyl_x2)
        
        self._cyl_y2 = create_spinbox(
            10.0, -1000, 1000, suffix=" mm"
        )
        cyl_layout.addRow("End Y:", self._cyl_y2)
        
        self._cyl_z2 = create_spinbox(
            10.0, -1000, 1000, suffix=" mm"
        )
        cyl_layout.addRow("End Z:", self._cyl_z2)
        
        self._cyl_r = create_spinbox(
            2.0, 0.1, 100, suffix=" mm"
        )
        cyl_layout.addRow("Radius:", self._cyl_r)
        
        self._add_cyl_btn = QPushButton("Add Cylinder")
        self._add_cyl_btn.clicked.connect(self._on_cylinder_clicked)
        cyl_layout.addRow(self._add_cyl_btn)
        
        layout.addWidget(cyl_group)
        
        # Add stretch
        layout.addStretch()
        
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def _on_sphere_clicked(self):
        center = (
            self._sphere_x.value(),
            self._sphere_y.value(),
            self._sphere_z.value()
        )
        radius = self._sphere_r.value()
        self.add_sphere.emit(center, radius)

    def _on_cylinder_clicked(self):
        start = (
            self._cyl_x1.value(),
            self._cyl_y1.value(),
            self._cyl_z1.value()
        )
        end = (
            self._cyl_x2.value(),
            self._cyl_y2.value(),
            self._cyl_z2.value()
        )
        radius = self._cyl_r.value()
        self.add_cylinder.emit(start, end, radius)
