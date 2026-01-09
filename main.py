"""
CT Simulation Application

A tool for generating simulated CT images from STL 3D models.

Features:
- Import STL mesh files
- Voxelize mesh with configurable resolution
- Simulate CT imaging with Hounsfield Units
- Export as DICOM series
"""

import sys


def main():
    """Application entry point."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from gui import MainWindow
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("CT Simulation")
    app.setOrganizationName("CT Simulator")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
