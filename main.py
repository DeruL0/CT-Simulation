"""
CT Simulation Software

Main entry point for the application.
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from config import DEFAULT_GUI
from gui.main_window import MainWindow
from gui.style import ScientificStyle
import logging


def setup_logging():
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def _resolve_dpi_policy(policy_name: str):
    """Map config string to Qt high-DPI rounding policy enum."""
    policy_map = {
        "PassThrough": Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
        "Round": Qt.HighDpiScaleFactorRoundingPolicy.Round,
        "Ceil": Qt.HighDpiScaleFactorRoundingPolicy.Ceil,
        "Floor": Qt.HighDpiScaleFactorRoundingPolicy.Floor,
        "RoundPreferFloor": Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor,
    }
    return policy_map.get(policy_name)


def _apply_dpi_policy_from_config() -> None:
    """Apply high-DPI rounding policy if configured."""
    policy_name = DEFAULT_GUI.high_dpi_rounding_policy
    if not policy_name:
        return

    policy = _resolve_dpi_policy(policy_name)
    if policy is None:
        logging.warning(
            "Unknown high_dpi_rounding_policy '%s'; using Qt default behavior.",
            policy_name,
        )
        return

    QApplication.setHighDpiScaleFactorRoundingPolicy(policy)



def main():
    """Application entry point."""
    setup_logging()

    # Optional DPI override from centralized config
    _apply_dpi_policy_from_config()

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(DEFAULT_GUI.application_name)
    app.setApplicationVersion(DEFAULT_GUI.application_version)
    app.setOrganizationName(DEFAULT_GUI.organization_name)

    # Set default font from centralized GUI config
    font = QFont(DEFAULT_GUI.font_family, DEFAULT_GUI.font_size)
    app.setFont(font)
    
    # Apply scientific white theme
    ScientificStyle.apply(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
