"""
UI layout assembly for MainWindow.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QGroupBox,
    QPushButton,
    QTabWidget,
    QScrollArea,
    QFrame,
    QStatusBar,
    QProgressBar,
)

from config import DEFAULT_GUI
from .panels import LoaderPanel, ParamsPanel, ViewerPanel, CompressionPanel
from .panels.structure_panel import StructurePanel
from .simulation_config import SimulationConfigBuilder
from visualization import MeshViewer


def setup_main_window_ui(window) -> None:
    """Build and wire main-window widgets and layout."""
    window.setWindowTitle(DEFAULT_GUI.window_title)
    window.setMinimumSize(*DEFAULT_GUI.min_size)
    window.resize(*DEFAULT_GUI.window_size)

    central = QWidget()
    window.setCentralWidget(central)
    main_layout = QHBoxLayout(central)
    main_layout.setContentsMargins(8, 8, 8, 8)
    main_layout.setSpacing(8)

    splitter = QSplitter(Qt.Horizontal)

    controls_widget = QWidget()
    controls_layout = QVBoxLayout(controls_widget)
    controls_layout.setContentsMargins(4, 4, 4, 4)

    window._loader_panel = LoaderPanel()
    controls_layout.addWidget(window._loader_panel)

    window._params_panel = ParamsPanel()
    controls_layout.addWidget(window._params_panel)

    window._structure_panel = StructurePanel(window._data_manager)
    controls_layout.addWidget(window._structure_panel)

    window._compression_panel = CompressionPanel(window._data_manager)
    controls_layout.addWidget(window._compression_panel)

    window._simulation_config_builder = SimulationConfigBuilder(
        params_panel=window._params_panel,
        loader_panel=window._loader_panel,
        structure_panel=window._structure_panel,
        compression_panel=window._compression_panel,
    )

    actions_group = QGroupBox("Actions")
    actions_layout = QVBoxLayout(actions_group)

    window._simulate_btn = QPushButton("Run Simulation")
    window._simulate_btn.setEnabled(False)
    window._simulate_btn.clicked.connect(window._on_simulate)
    actions_layout.addWidget(window._simulate_btn)

    window._export_btn = QPushButton("Export DICOM")
    window._export_btn.setObjectName("secondaryButton")
    window._export_btn.setEnabled(False)
    window._export_btn.clicked.connect(window._on_export)
    actions_layout.addWidget(window._export_btn)

    window._reset_stl_btn = QPushButton("Reset to STL")
    window._reset_stl_btn.setObjectName("secondaryButton")
    window._reset_stl_btn.setEnabled(False)
    window._reset_stl_btn.clicked.connect(window._on_reset_stl)
    actions_layout.addWidget(window._reset_stl_btn)

    controls_layout.addWidget(actions_group)
    controls_layout.addStretch()

    scroll_area = QScrollArea()
    scroll_area.setWidget(controls_widget)
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.NoFrame)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll_area.setMinimumWidth(380)
    scroll_area.setMaximumWidth(550)

    splitter.addWidget(scroll_area)

    viewer_tabs = QTabWidget()
    window._viewer_3d_panel = MeshViewer()
    viewer_tabs.addTab(window._viewer_3d_panel, "3D View")

    window._viewer_panel = ViewerPanel()
    viewer_tabs.addTab(window._viewer_panel, "2D Slices")

    splitter.addWidget(viewer_tabs)

    splitter.setCollapsible(0, False)
    splitter.setCollapsible(1, False)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([420, 980])
    main_layout.addWidget(splitter)

    window._status_bar = QStatusBar()
    window.setStatusBar(window._status_bar)

    window._progress_bar = QProgressBar()
    window._progress_bar.setMaximumWidth(200)
    window._progress_bar.setVisible(False)
    window._status_bar.addPermanentWidget(window._progress_bar)

    window._status_bar.showMessage("Ready")
