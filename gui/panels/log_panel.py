"""
Log Viewer Panel

Provides a widget to display application logs in real-time.
"""

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, 
    QPushButton, QComboBox, QLabel
)
from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QFont, QColor


class QLogHandler(logging.Handler):
    """
    Custom logging handler that emits a signal for each log record.
    """
    
    def __init__(self, parent=None):
        super().__init__()
        self.emitter = LogEmitter(parent)
        
    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_message.emit(msg, record.levelno)


class LogEmitter(QObject):
    """Helper object to emit signals from the logging handler."""
    log_message = Signal(str, int)


class LogViewerPanel(QWidget):
    """Panel for viewing application logs."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_logging()
        
    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 0)
        
        toolbar.addWidget(QLabel("Logs"))
        
        self._level_combo = QComboBox()
        self._level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._level_combo.setCurrentText("INFO")
        self._level_combo.currentTextChanged.connect(self._on_level_changed)
        toolbar.addWidget(self._level_combo)
        
        toolbar.addStretch()
        
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self.clear_logs)
        toolbar.addWidget(clear_btn)
        
        layout.addLayout(toolbar)
        
        # Log text area
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
        
        # Monospace font
        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Monospace", 9)
        self._text_edit.setFont(font)
        
        # Style
        self._text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #FAFAFA;
                color: #333333;
                border: 1px solid #DDDDDD;
            }
        """)
        
        layout.addWidget(self._text_edit)
        
    def _setup_logging(self) -> None:
        """Set up the log handler."""
        self._handler = QLogHandler(self)
        self._handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        connections = self._handler.emitter.log_message.connect(self._append_log)
        
        # Add to root logger
        logging.getLogger().addHandler(self._handler)
        logging.getLogger().setLevel(logging.INFO)
        
    def _on_level_changed(self, text: str) -> None:
        """Handle log level change."""
        level = getattr(logging, text)
        logging.getLogger().setLevel(level)
        self.log_info(f"Log level set to {text}")
        
    @Slot(str, int)
    def _append_log(self, msg: str, levelno: int) -> None:
        """Append a log message to the text area."""
        color = "#000000"
        if levelno >= logging.ERROR:
            color = "#D32F2F"  # Red
        elif levelno >= logging.WARNING:
            color = "#F57C00"  # Orange
        elif levelno >= logging.INFO:
            color = "#1976D2"  # Blue
        elif levelno >= logging.DEBUG:
            color = "#757575"  # Grey
            
        html = f'<span style="color:{color};">{msg}</span>'
        self._text_edit.append(html)
        
    def clear_logs(self) -> None:
        """Clear the log display."""
        self._text_edit.clear()
        
    def log_info(self, msg: str) -> None:
        """Helper to log info."""
        logging.info(msg)
