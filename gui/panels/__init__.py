"""
GUI Panels Package

Contains all panel widgets for the application.
"""

from .loader_panel import LoaderPanel
from .params_panel import ParamsPanel
from .viewer_panel import ViewerPanel
from .log_panel import LogViewerPanel
from .compression_panel import CompressionPanel

# Alias for consistency
LogPanel = LogViewerPanel
STLPanel = LoaderPanel  # Alias for backward compatibility

__all__ = [
    'LoaderPanel',
    'STLPanel',
    'ParamsPanel', 
    'ViewerPanel',
    'LogViewerPanel',
    'LogPanel',
    'CompressionPanel',
]
