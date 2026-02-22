"""
Shared UI scaffold for settings-like panel pages.
"""

from PySide6.QtWidgets import QFormLayout, QSizePolicy, QVBoxLayout, QWidget


class BaseSettingsPage(QWidget):
    """
    Base page providing common layout boilerplate for panel sub-pages.
    """

    CONTENT_MARGINS = (0, 0, 0, 0)

    def _create_main_layout(self) -> QVBoxLayout:
        """
        Create standardized top-level layout and size policy.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*self.CONTENT_MARGINS)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        return layout

    @staticmethod
    def _create_form_layout() -> QFormLayout:
        """
        Create a form layout for setting rows.
        """
        return QFormLayout()

    @staticmethod
    def _finalize_layout(layout: QVBoxLayout) -> None:
        """
        Apply standard trailing stretch so controls stay top-aligned.
        """
        layout.addStretch()
