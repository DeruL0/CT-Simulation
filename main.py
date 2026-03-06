"""
CT Simulation Software

Main entry point for the application.
"""

import atexit
import faulthandler
import logging
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from config import DEFAULT_GUI
from gui.main_window import MainWindow
from gui.style import ScientificStyle


_CRASH_LOG_STREAM = None
_RUNTIME_CLEANED = False


def setup_logging():
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def setup_crash_diagnostics() -> None:
    """Enable faulthandler logging for native crashes (access violations, aborts)."""
    global _CRASH_LOG_STREAM
    try:
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        crash_log_path = log_dir / "native_crash.log"
        _CRASH_LOG_STREAM = crash_log_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_CRASH_LOG_STREAM, all_threads=True)
        logging.info("Native crash diagnostics enabled: %s", crash_log_path)
    except (OSError, RuntimeError, ValueError) as exc:
        logging.warning("Failed to initialize native crash log file: %s", exc)
        try:
            faulthandler.enable(all_threads=True)
        except RuntimeError:
            logging.warning("faulthandler is unavailable in this runtime.")


def _cleanup_gpu_runtime() -> None:
    """
    Best-effort GPU runtime cleanup.

    This avoids leaving asynchronous CUDA work and pooled allocations alive while
    the interpreter tears down native modules.
    """
    try:
        import cupy as cp
    except Exception:
        return

    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass

    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def cleanup_runtime() -> None:
    """Idempotent process shutdown cleanup."""
    global _RUNTIME_CLEANED, _CRASH_LOG_STREAM
    if _RUNTIME_CLEANED:
        return
    _RUNTIME_CLEANED = True

    _cleanup_gpu_runtime()

    if _CRASH_LOG_STREAM is not None:
        try:
            _CRASH_LOG_STREAM.flush()
            _CRASH_LOG_STREAM.close()
        except OSError:
            pass
        _CRASH_LOG_STREAM = None


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
    setup_crash_diagnostics()
    atexit.register(cleanup_runtime)

    # Optional DPI override from centralized config
    _apply_dpi_policy_from_config()

    # Create application
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(cleanup_runtime)
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
