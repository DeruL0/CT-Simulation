"""
Scientific White Theme Stylesheet

Provides a clean, professional Qt stylesheet for scientific applications.
Uses a white background with dark text and subtle accent colors.
"""

# Scientific white theme color palette
COLORS = {
    "background": "#FFFFFF",
    "background_alt": "#F8F9FA",
    "surface": "#FFFFFF",
    "border": "#E0E0E0",
    "border_focus": "#2962FF",
    "text": "#333333",
    "text_secondary": "#666666",
    "text_disabled": "#999999",
    "accent": "#2962FF",
    "accent_light": "#E3F2FD",
    "accent_hover": "#1E88E5",
    "accent_pressed": "#1565C0",
    "success": "#43A047",
    "warning": "#FB8C00",
    "error": "#E53935",
    "divider": "#EEEEEE",
}

# Font settings
FONTS = {
    "family": "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif",
    "size": "10pt",
    "size_small": "9pt",
    "size_large": "11pt",
    "size_header": "12pt",
}


def get_stylesheet() -> str:
    """Get the complete Qt stylesheet for the scientific white theme."""
    return f"""
    /* ============================================ */
    /*            GLOBAL STYLES                     */
    /* ============================================ */
    
    QWidget {{
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
        font-family: {FONTS["family"]};
        font-size: {FONTS["size"]};
    }}
    
    QMainWindow {{
        background-color: {COLORS["background"]};
    }}
    
    /* ============================================ */
    /*            MENU BAR                          */
    /* ============================================ */
    
    QMenuBar {{
        background-color: {COLORS["surface"]};
        border-bottom: 1px solid {COLORS["border"]};
        padding: 4px 0;
    }}
    
    QMenuBar::item {{
        padding: 6px 12px;
        background: transparent;
        border-radius: 4px;
        margin: 0 2px;
    }}
    
    QMenuBar::item:selected {{
        background-color: {COLORS["accent_light"]};
        color: {COLORS["accent"]};
    }}
    
    QMenu {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        padding: 4px;
    }}
    
    QMenu::item {{
        padding: 8px 32px 8px 16px;
        border-radius: 4px;
    }}
    
    QMenu::item:selected {{
        background-color: {COLORS["accent_light"]};
        color: {COLORS["accent"]};
    }}
    
    QMenu::separator {{
        height: 1px;
        background: {COLORS["divider"]};
        margin: 4px 8px;
    }}
    
    /* ============================================ */
    /*            BUTTONS                           */
    /* ============================================ */
    
    QPushButton {{
        background-color: {COLORS["accent"]};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS["accent_hover"]};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS["accent_pressed"]};
    }}
    
    QPushButton:disabled {{
        background-color: {COLORS["border"]};
        color: {COLORS["text_disabled"]};
    }}
    
    QPushButton[flat="true"], QPushButton#secondaryButton {{
        background-color: transparent;
        color: {COLORS["accent"]};
        border: 1px solid {COLORS["accent"]};
    }}
    
    QPushButton[flat="true"]:hover, QPushButton#secondaryButton:hover {{
        background-color: {COLORS["accent_light"]};
    }}
    
    /* ============================================ */
    /*            INPUT FIELDS                      */
    /* ============================================ */
    
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        padding: 6px 10px;
        selection-background-color: {COLORS["accent_light"]};
        selection-color: {COLORS["accent"]};
    }}
    
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, 
    QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {COLORS["accent"]};
    }}
    
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background-color: {COLORS["background_alt"]};
        color: {COLORS["text_disabled"]};
    }}
    
    /* ============================================ */
    /*            COMBO BOX                         */
    /* ============================================ */
    
    QComboBox {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        padding: 6px 10px;
        min-width: 120px;
    }}
    
    QComboBox:focus {{
        border-color: {COLORS["accent"]};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {COLORS["text_secondary"]};
        margin-right: 8px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        selection-background-color: {COLORS["accent_light"]};
        selection-color: {COLORS["accent"]};
    }}
    
    /* ============================================ */
    /*            SLIDERS                           */
    /* ============================================ */
    
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: {COLORS["border"]};
        border-radius: 2px;
    }}
    
    QSlider::handle:horizontal {{
        background: {COLORS["accent"]};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {COLORS["accent_hover"]};
    }}
    
    QSlider::sub-page:horizontal {{
        background: {COLORS["accent"]};
        border-radius: 2px;
    }}
    
    /* ============================================ */
    /*            PROGRESS BAR                      */
    /* ============================================ */
    
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background: {COLORS["background_alt"]};
        text-align: center;
        height: 8px;
    }}
    
    QProgressBar::chunk {{
        background: {COLORS["accent"]};
        border-radius: 4px;
    }}
    
    /* ============================================ */
    /*            GROUP BOX                         */
    /* ============================================ */
    
    QGroupBox {{
        font-weight: 600;
        font-size: {FONTS["size_large"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
    }}
    
    /* ============================================ */
    /*            LABELS                            */
    /* ============================================ */
    
    QLabel {{
        color: {COLORS["text"]};
        background: transparent;
    }}
    
    QLabel#headerLabel {{
        font-size: {FONTS["size_header"]};
        font-weight: 600;
        color: {COLORS["text"]};
    }}
    
    QLabel#secondaryLabel {{
        color: {COLORS["text_secondary"]};
        font-size: {FONTS["size_small"]};
    }}
    
    /* ============================================ */
    /*            SCROLL AREA                       */
    /* ============================================ */
    
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    
    QScrollBar:vertical {{
        background: {COLORS["background_alt"]};
        width: 10px;
        border-radius: 5px;
        margin: 0;
    }}
    
    QScrollBar::handle:vertical {{
        background: {COLORS["border"]};
        border-radius: 5px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {COLORS["text_secondary"]};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    
    QScrollBar:horizontal {{
        background: {COLORS["background_alt"]};
        height: 10px;
        border-radius: 5px;
        margin: 0;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {COLORS["border"]};
        border-radius: 5px;
        min-width: 30px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: {COLORS["text_secondary"]};
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    
    /* ============================================ */
    /*            TAB WIDGET                        */
    /* ============================================ */
    
    QTabWidget::pane {{
        border: 1px solid {COLORS["border"]};
        border-radius: 4px;
        background-color: {COLORS["surface"]};
    }}
    
    QTabBar::tab {{
        background-color: transparent;
        border: none;
        padding: 10px 20px;
        margin-right: 2px;
        color: {COLORS["text_secondary"]};
    }}
    
    QTabBar::tab:selected {{
        color: {COLORS["accent"]};
        border-bottom: 2px solid {COLORS["accent"]};
    }}
    
    QTabBar::tab:hover:!selected {{
        color: {COLORS["text"]};
        background-color: {COLORS["background_alt"]};
    }}
    
    /* ============================================ */
    /*            SPLITTER                          */
    /* ============================================ */
    
    QSplitter::handle {{
        background-color: {COLORS["divider"]};
    }}
    
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    
    /* ============================================ */
    /*            STATUS BAR                        */
    /* ============================================ */
    
    QStatusBar {{
        background-color: {COLORS["background_alt"]};
        border-top: 1px solid {COLORS["border"]};
        color: {COLORS["text_secondary"]};
    }}
    
    QStatusBar::item {{
        border: none;
    }}
    
    /* ============================================ */
    /*            TOOL BAR                          */
    /* ============================================ */
    
    QToolBar {{
        background-color: {COLORS["surface"]};
        border-bottom: 1px solid {COLORS["border"]};
        spacing: 4px;
        padding: 4px;
    }}
    
    QToolButton {{
        background-color: transparent;
        border: none;
        border-radius: 4px;
        padding: 6px;
    }}
    
    QToolButton:hover {{
        background-color: {COLORS["accent_light"]};
    }}
    
    QToolButton:pressed {{
        background-color: {COLORS["accent"]};
        color: white;
    }}
    
    /* ============================================ */
    /*            CHECKBOX & RADIO                  */
    /* ============================================ */
    
    QCheckBox, QRadioButton {{
        spacing: 8px;
    }}
    
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px;
        height: 18px;
    }}
    
    QCheckBox::indicator {{
        border: 2px solid {COLORS["border"]};
        border-radius: 3px;
        background: {COLORS["surface"]};
    }}
    
    QCheckBox::indicator:checked {{
        background: {COLORS["accent"]};
        border-color: {COLORS["accent"]};
    }}
    
    QRadioButton::indicator {{
        border: 2px solid {COLORS["border"]};
        border-radius: 9px;
        background: {COLORS["surface"]};
    }}
    
    QRadioButton::indicator:checked {{
        background: {COLORS["accent"]};
        border-color: {COLORS["accent"]};
    }}
    """


class ScientificStyle:
    """Helper class for applying the scientific white theme."""
    
    @staticmethod
    def apply(app) -> None:
        """
        Apply the scientific white theme to a QApplication.
        
        Args:
            app: QApplication instance
        """
        app.setStyleSheet(get_stylesheet())
    
    @staticmethod
    def get_color(name: str) -> str:
        """Get a color value by name."""
        return COLORS.get(name, COLORS["text"])
    
    @staticmethod
    def get_font_size(name: str) -> str:
        """Get a font size by name."""
        return FONTS.get(name, FONTS["size"])
