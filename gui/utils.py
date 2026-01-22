"""
GUI Utilities

Helper functions for creating common UI elements and managing layouts
to reduce code duplication across panels.
"""

from typing import Union, Callable, Optional
from PySide6.QtWidgets import (
    QSpinBox, QDoubleSpinBox, QStackedWidget, 
    QSizePolicy, QWidget
)

def create_spinbox(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    step: Union[int, float] = 1.0,
    decimals: int = 0,
    suffix: str = "",
    tooltip: str = "",
    callback: Optional[Callable] = None,
    parent: Optional[QWidget] = None
) -> Union[QSpinBox, QDoubleSpinBox]:
    """
    Create and configure a QSpinBox or QDoubleSpinBox.
    
    Args:
        value: Initial value
        min_val: Minimum value
        max_val: Maximum value
        step: Step size
        decimals: Number of decimals (0 for integer QSpinBox)
        suffix: Suffix string (e.g. " mm")
        tooltip: Tooltip text
        callback: Function to call on valueChanged
        parent: Parent widget
        
    Returns:
        Configured spinbox
    """
    if isinstance(value, float) or decimals > 0:
        spin = QDoubleSpinBox(parent)
        spin.setDecimals(decimals)
    else:
        spin = QSpinBox(parent)
    
    spin.setRange(min_val, max_val)
    spin.setValue(value)
    spin.setSingleStep(step)
    
    if suffix:
        spin.setSuffix(suffix)
    if tooltip:
        spin.setToolTip(tooltip)
    if callback:
        spin.valueChanged.connect(callback)
        
    return spin

def update_stack_sizing(stack: QStackedWidget, index: int) -> None:
    """
    Update stack widget size policy to fit current page.
    
    Commonly used to make QStackedWidget resize to fit its content
    instead of taking the size of the largest page.
    
    Args:
        stack: The QStackedWidget to update
        index: The index of the visible page
    """
    for i in range(stack.count()):
        widget = stack.widget(i)
        if i == index:
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            widget.updateGeometry()
        else:
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    stack.adjustSize()
