"""
GUI Panels Package

Contains all panel widgets for the application.
"""

from .stl_panel import STLPanel
from .params_panel import ParamsPanel
from .viewer_panel import ViewerPanel
from .log_panel import LogViewerPanel

# Alias for consistency
LogPanel = LogViewerPanel

__all__ = [
    'STLPanel',
    'ParamsPanel', 
    'ViewerPanel',
    'LogViewerPanel',
    'LogPanel',
]
