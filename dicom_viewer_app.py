"""
Visualiseur DICOM multi-vues — application autonome.

Entrée 1 (requise)     : séquence dynamique ou volume 3D DICOM
                          (fichier DICOM multi-trame OU dossier contenant une
                          coupe DICOM par fichier).
Entrée 2 (optionnelle) : image sagittale de référence DICOM (2D).

Sortie : fenêtre GUI à 4 panneaux — AXIAL, CORONAL, SAGITTAL (coupes du
volume, navigables par sliders/molette) et RÉFÉRENCE SAGITTALE (image 2D
fournie, affichée telle quelle).

Basé sur MITKviewer.py : même squelette de fenêtre (sidebar + grille 2×2 de
canvases matplotlib), mêmes conventions de sliders/molette/reset. Le 4e
panneau, inutilisé dans MITKviewer.py (étiqueté "3d"), est ici dédié à
l'image de référence. Le chargement DICOM et le mode autonome (dialogues de
fichiers, CLI) sont nouveaux : MITKviewer.py était conçu pour être appelé par
un pipeline avec des tableaux numpy déjà en mémoire.

Usage :
    python dicom_viewer_app.py [volume] [reference]

    volume    : fichier DICOM multi-trame ou dossier de coupes (optionnel :
                sinon, à charger depuis l'interface, par clic ou par
                glisser-déposer sur le bouton Volume).
    reference : fichier DICOM 2D de référence (optionnel).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QMessageBox, QPushButton, QFileDialog, QMenu,
    QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, QTimer, Signal, QUrl, QThread
from PySide6.QtGui import QDesktopServices, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

_COLOR_AXIAL = "#FF5555"
_COLOR_CORONAL = "#5555FF"
_COLOR_SAGITTAL = "#55FF55"
_COLOR_REFERENCE = "#FFD166"

_STATUS_STYLE_NORMAL = "color: #999999; font-size: 13px; margin-top: 15px;"
_STATUS_STYLE_LOADING = "color: #00E5FF; font-size: 13px; font-weight: bold; margin-top: 15px;"


def _resource_path(name: str) -> Path:
    """Résout un fichier embarqué (ex. icon.ico), que l'app tourne comme
    script ou comme exécutable PyInstaller (sys._MEIPASS)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


# ── Chargement DICOM ─────────────────────────────────────────────────────────

def _read_dicom(path: Path) -> pydicom.dataset.FileDataset:
    try:
        return pydicom.dcmread(str(path), force=False)
    except InvalidDicomError:
        return pydicom.dcmread(str(path), force=True)


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Réduit un tableau couleur (..., 3|4) à un seul canal (luminosité)."""
    if arr.ndim >= 1 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float64)
        return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return arr


def _extract_spacing(ds) -> "tuple[float, float] | None":
    """Retourne (row_spacing_cm, col_spacing_cm) depuis les tags DICOM usuels."""
    sp = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing") or ds.get("NominalScannedPixelSpacing")
    if sp is None:
        try:
            sp = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        except Exception:
            pass
    if sp is None:
        try:
            sp = ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        except Exception:
            pass
    if sp is None:
        # Repli échographie : régions US portent leur propre delta physique.
        try:
            region = ds.SequenceOfUltrasoundRegions[0]
            dx = float(getattr(region, "PhysicalDeltaX", 0))
            dy = float(getattr(region, "PhysicalDeltaY", dx))
            if dx > 0:
                sp = [dy, dx]
        except Exception:
            pass
    if sp is None:
        return None
    try:
        return float(sp[0]), float(sp[1])
    except Exception:
        return None


def _spacing_str(sp) -> str:
    if sp is None:
        return "spacing ?"
    return f"{sp[0]:.3g} × {sp[1]:.3g} cm"


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def load_volume_from_file(path: Path):
    """Charge un volume 3D depuis un fichier DICOM multi-trame unique.

    Retourne (volume (Z,Y,X) float64, spacing_yx | None).
    """
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Chargement du volume/séquence « {path.name} » ({size_mb:.0f} Mo)…")
    ds = _read_dicom(path)
    arr = _to_grayscale(np.asarray(ds.pixel_array))
    if arr.ndim == 2:
        raise ValueError(
            f"« {path.name} » ne contient qu'une seule image (2D) : ce n'est "
            f"pas une séquence/volume. Utilisez un dossier de coupes ou un "
            f"fichier DICOM multi-trame, ou chargez ce fichier comme image "
            f"de référence."
        )
    if arr.ndim != 3:
        raise ValueError(f"Forme de pixel_array inattendue {arr.shape} dans « {path.name} ».")

    arr = np.transpose(arr, (1, 0, 2))
    return arr.astype(np.float64), _extract_spacing(ds)


def load_volume_from_folder(folder: Path):
    """Charge une série DICOM (une coupe par fichier) depuis un dossier.

    Retourne (volume (Z,Y,X) float64, spacing_yx | None).
    """
    if sitk is not None:
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(str(folder))
        if names:
            reader.SetFileNames(names)
            img = reader.Execute()
            arr = _to_grayscale(sitk.GetArrayFromImage(img).astype(np.float64))  # (Z,Y,X)
            sx, sy, _sz = img.GetSpacing()
            return arr, (sy, sx)

    # Repli sans SimpleITK (ou dossier non reconnu comme série GDCM).
    files = [p for p in sorted(folder.iterdir()) if p.is_file()]
    if not files:
        raise ValueError(f"Aucun fichier DICOM valide trouvé dans « {folder} ».")
    return load_volume_from_file_list(files)


def load_volume_from_file_list(paths: "list[Path]"):
    """Charge une série DICOM à partir d'une liste explicite de fichiers (une coupe
    chacun), triés par InstanceNumber. Utilisé en repli pour un dossier scanné, et
    pour plusieurs fichiers déposés ensemble par glisser-déposer.

    Retourne (volume (Z,Y,X) float64, spacing_yx | None).
    """
    datasets = []
    for p in paths:
        try:
            datasets.append(_read_dicom(p))
        except Exception:
            continue
    if not datasets:
        raise ValueError("Aucun fichier DICOM valide dans la sélection.")

    datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    slices = [_to_grayscale(np.asarray(d.pixel_array)) for d in datasets]
    if len({s.shape for s in slices}) != 1:
        raise ValueError("Les coupes DICOM sélectionnées n'ont pas toutes la même taille.")
    volume = np.stack(slices, axis=0).astype(np.float64)
    return volume, _extract_spacing(datasets[0])


def load_reference_image(path: Path):
    """Charge une image DICOM 2D unique.

    Retourne (image (H,W) float64, spacing_yx | None).
    """
    ds = _read_dicom(path)
    arr = _to_grayscale(np.asarray(ds.pixel_array))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"« {path.name} » n'est pas une image 2D unique (forme {arr.shape}). "
            f"L'image de référence doit être une seule coupe sagittale."
        )
    return arr.astype(np.float64), _extract_spacing(ds)


# ── Bouton avec glisser-déposer ──────────────────────────────────────────────

class _DropButton(QPushButton):
    """Bouton acceptant le glisser-déposer d'un ou plusieurs fichiers/dossiers,
    en plus de son comportement normal au clic (menu ou action directe)."""

    files_dropped = Signal(list)

    _STYLE = (
        "QPushButton {{ background-color: {bg}; color: white; padding: 14px; "
        "font-size: 15px; border-radius: 4px; border: {border}; text-align: left; }}"
        "QPushButton:hover {{ background-color: #2a2a3e; }}"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self._set_drag_active(False)

    def _set_drag_active(self, active: bool):
        border = "2px dashed #00E5FF" if active else "1px solid #333"
        bg = "#26364a" if active else "#1e1e2e"
        self.setStyleSheet(self._STYLE.format(bg=bg, border=border))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._set_drag_active(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._set_drag_active(False)

    def dropEvent(self, event):
        self._set_drag_active(False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


# ── Chargement en arrière-plan ───────────────────────────────────────────────

class _LoadThread(QThread):
    """Exécute une fonction de chargement (lecture DICOM + normalisation) hors
    du thread GUI, pour qu'une grosse séquence ne fige pas toute l'interface
    (barre de titre "Ne répond pas", fenêtre grisée par Windows) pendant sa
    décompression/conversion."""

    succeeded = Signal(object, object)  # (array, spacing)
    failed = Signal(str)

    def __init__(self, loader_fn, parent=None):
        super().__init__(parent)
        self._loader_fn = loader_fn

    def run(self):
        try:
            arr, spacing = self._loader_fn()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(arr, spacing)


# ── Fenêtre principale ───────────────────────────────────────────────────────

class DicomViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualiseur DICOM Multi-Vues — Volume + Référence Sagittale")
        self.setWindowIcon(QIcon(str(_resource_path("icon.ico"))))
        self.setStyleSheet("background-color: #121212; color: white; font-size: 15px;")

        self.volume: "np.ndarray | None" = None      # (Z,Y,X) uint8
        self.volume_spacing = None
        self.reference: "np.ndarray | None" = None    # (H,W) uint8
        self.reference_spacing = None

        self._plot_artists = {}
        self._ref_title_artists = []
        self._update_pending = False
        self._closing = False

        # Identifie, parmi l'historique du panneau "FICHIERS CHARGÉS", quelle
        # entrée correspond à ce qui est actuellement affiché : un retrait ne
        # doit effacer l'affichage que si l'entrée retirée est bien celle-là
        # (une entrée plus ancienne, déjà remplacée par un chargement suivant,
        # ne doit faire disparaître aucun affichage puisqu'elle n'en a plus).
        self._load_counter = 0
        self._current_volume_load_id = None
        self._current_reference_load_id = None

        # Chargements en cours (threads), pour ne pas geler l'IHM (cf. _LoadThread).
        self._volume_thread: "_LoadThread | None" = None
        self._reference_thread: "_LoadThread | None" = None

        self._build_ui()
        self.showMaximized()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- SIDEBAR ---
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(15, 20, 15, 20)

        title_load = QLabel("CHARGEMENT")
        title_load.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #AAAAAA;")
        sidebar.addWidget(title_load)

        btn_vol = self.btn_vol = _DropButton("Volume : fichier ou dossier…")
        btn_vol.setToolTip(
            "Séquence/volume DICOM : fichier multi-trame ou dossier de coupes.\n"
            "Cliquez pour choisir, ou glissez-déposez directement le fichier/dossier ici."
        )
        menu_vol = QMenu(btn_vol)
        menu_vol.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: white;
                border: 1px solid #444;
                padding: 4px;
            }
            QMenu::item {
                padding: 12px 28px;
                font-size: 14px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #00E5FF;
                color: #000000;
            }
        """)
        menu_vol.addAction("Fichier DICOM (multi-trame)…", self.action_load_volume_file)
        menu_vol.addAction("Dossier (série de coupes)…", self.action_load_volume_folder)
        btn_vol.setMenu(menu_vol)
        btn_vol.files_dropped.connect(self.on_volume_dropped)
        sidebar.addWidget(btn_vol)

        btn_ref = self.btn_ref = _DropButton("Image sagittale de référence…")
        btn_ref.setToolTip(
            "Image DICOM 2D unique utilisée comme référence anatomique (optionnelle).\n"
            "Cliquez pour choisir, ou glissez-déposez directement le fichier ici."
        )
        btn_ref.clicked.connect(self.action_load_reference)
        btn_ref.files_dropped.connect(self.on_reference_dropped)
        sidebar.addWidget(btn_ref)

        sidebar.addSpacing(20)

        title_nav = QLabel("NAVIGATION")
        title_nav.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #AAAAAA;")
        sidebar.addWidget(title_nav)

        self.slider_ax, self.lbl_ax = self.create_mitk_slider("AXIAL", 1, sidebar, _COLOR_AXIAL)
        self.slider_cor, self.lbl_cor = self.create_mitk_slider("CORONAL", 1, sidebar, _COLOR_CORONAL)
        self.slider_sag, self.lbl_sag = self.create_mitk_slider("SAGITTAL", 1, sidebar, _COLOR_SAGITTAL)
        for s in (self.slider_ax, self.slider_cor, self.slider_sag):
            s.setEnabled(False)

        sidebar.addSpacing(20)

        self.btn_reset = QPushButton("RESET VIEWS")
        self.btn_reset.setMinimumHeight(58)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #00E5FF; color: #000000; font-weight: bold;
                font-size: 17px; border-radius: 8px; border: 2px solid #00B8D4;
            }
            QPushButton:hover { background-color: #64FFDA; }
        """)
        self.btn_reset.clicked.connect(self.reset_views)
        sidebar.addWidget(self.btn_reset)

        sidebar.addSpacing(10)

        self.btn_clear = QPushButton("EFFACER TOUT")
        self.btn_clear.setMinimumHeight(48)
        self.btn_clear.setToolTip("Efface le volume et l'image de référence chargés.")
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #3a1e1e; color: #FF8A8A; font-weight: bold;
                font-size: 15px; border-radius: 8px; border: 2px solid #FF6B6B;
            }
            QPushButton:hover { background-color: #4a2626; }
        """)
        self.btn_clear.clicked.connect(self.clear_all)
        sidebar.addWidget(self.btn_clear)

        self.lbl_status = QLabel("Volume : non chargé\nRéférence : non chargée ")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(_STATUS_STYLE_NORMAL)
        sidebar.addWidget(self.lbl_status)

        sidebar.addSpacing(15)

        title_history = QLabel("FICHIERS CHARGÉS")
        title_history.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 6px; color: #AAAAAA;")
        sidebar.addWidget(title_history)

        self.list_history = QListWidget()
        self.list_history.setToolTip("Clic droit : ouvrir le dossier ou retirer de la liste.")
        self.list_history.setStyleSheet("""
            QListWidget {
                background-color: #0d0d0d; color: #dddddd; border: 1px solid #333;
                border-radius: 4px; font-size: 14px; font-weight: bold;
            }
            QListWidget::item { padding: 7px 5px; border-bottom: 1px solid #222; }
        """)
        self.list_history.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_history.customContextMenuRequested.connect(self._show_history_context_menu)
        sidebar.addWidget(self.list_history, 1)

        main_layout.addLayout(sidebar, 1)

        # --- GRILLE DES VUES ---
        grid_layout = QVBoxLayout()
        row1, row2 = QHBoxLayout(), QHBoxLayout()

        self.canvas_ax, self.ax_ax = self.create_view_canvas("axial")
        self.canvas_sag, self.ax_sag = self.create_view_canvas("sagittal")
        self.canvas_cor, self.ax_cor = self.create_view_canvas("coronal")
        self.canvas_ref, self.ax_ref = self.create_view_canvas("reference")

        row1.addWidget(self.canvas_ax); row1.addWidget(self.canvas_sag)
        row2.addWidget(self.canvas_cor); row2.addWidget(self.canvas_ref)

        grid_layout.addLayout(row1); grid_layout.addLayout(row2)
        main_layout.addLayout(grid_layout, 5)

        for ax in (self.ax_ax, self.ax_sag, self.ax_cor):
            self._draw_placeholder(ax, "Aucun volume chargé")
        self._render_reference()

    def create_mitk_slider(self, name, max_val, layout, color):
        lbl = QLabel(f"{name} (Slice 1 / {max_val})")
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 16px; margin-top: 15px;")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimumHeight(28)
        slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #333; border-radius: 4px; }
            QSlider::handle:horizontal {
                width: 22px; height: 22px; margin: -8px 0;
                background: #00E5FF; border-radius: 11px;
            }
        """)
        slider.setRange(0, max_val - 1)
        slider.setValue(max_val // 2)
        slider.valueChanged.connect(self.schedule_update)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider, lbl

    def create_view_canvas(self, view_type):
        fig = Figure(facecolor='black')
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.axis('off')
        if view_type in ("axial", "coronal", "sagittal"):
            canvas.wheelEvent = lambda event: self.handle_scroll(event, view_type)
        return canvas, ax

    def _draw_placeholder(self, ax, text):
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')
        if text:
            ax.text(0.5, 0.5, text, transform=ax.transAxes, color="#555555",
                     ha='center', va='center', fontsize=16)

    # ------------------------------------------------------------------
    # Chargement (actions UI)
    # ------------------------------------------------------------------

    def action_load_volume_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un fichier DICOM (séquence/volume)", "",
            "Tous les fichiers (*);;DICOM (*.dcm *.dicom)")
        if path:
            self.load_volume_path(Path(path))

    def action_load_volume_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choisir un dossier DICOM (série de coupes)")
        if folder:
            self.load_volume_path(Path(folder))

    def action_load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir l'image sagittale de référence (DICOM)", "",
            "Tous les fichiers (*);;DICOM (*.dcm *.dicom)")
        if path:
            self.load_reference_path(Path(path))

    def on_reference_dropped(self, raw_paths: "list[str]"):
        """Glisser-déposer sur le bouton de référence : une seule image 2D est
        attendue, donc seul le premier fichier déposé est utilisé."""
        self.load_reference_path(Path(raw_paths[0]))

    def load_volume_path(self, path: Path):
        """Charge le volume depuis un chemin unique (fichier multi-trame ou dossier)."""
        if path.is_dir():
            self._load_volume_common(lambda: load_volume_from_folder(path), path.name, path)
        else:
            self._load_volume_common(lambda: load_volume_from_file(path), path.name, path.parent)

    def load_volume_file_list(self, paths: "list[Path]"):
        """Charge le volume depuis plusieurs fichiers déposés ensemble (une série)."""
        self._load_volume_common(lambda: load_volume_from_file_list(paths),
                                  f"{len(paths)} fichiers (série)", paths[0].parent)

    def on_volume_dropped(self, raw_paths: "list[str]"):
        """Glisser-déposer sur le bouton volume.

        Un seul élément (fichier ou dossier) suit le même chemin que le clic ;
        plusieurs fichiers déposés ensemble sont traités comme les coupes d'une
        série (les dossiers éventuellement mêlés dans la sélection sont ignorés).
        """
        paths = [Path(p) for p in raw_paths]
        if len(paths) == 1:
            self.load_volume_path(paths[0])
            return
        files = [p for p in paths if p.is_file()]
        if not files:
            QMessageBox.warning(self, "Glisser-déposer",
                                 "Aucun fichier valide dans la sélection déposée.")
            return
        self.load_volume_file_list(files)

    def _load_volume_common(self, loader_fn, source_label, directory: Path):
        # Chargé dans un thread (_LoadThread) : une séquence dynamique peut
        # peser plusieurs dizaines/centaines de Mo à décoder + normaliser, et un
        # appel bloquant sur le thread GUI gèle toute l'interface pendant ce
        # temps (Windows affiche "Ne répond pas"). Le thread laisse la fenêtre
        # réactive ; le rendu réel n'a lieu qu'au retour sur le thread GUI,
        # dans _on_volume_loaded (slot connecté à un signal -> appelé en
        # sécurité côté GUI même si _LoadThread tourne ailleurs).
        if self._volume_thread is not None:
            return  # un chargement de volume est déjà en cours

        self._set_loading_state(True)
        thread = _LoadThread(loader_fn, self)
        thread.succeeded.connect(
            lambda arr, spacing: self._on_volume_loaded(arr, spacing, source_label, directory))
        thread.failed.connect(self._on_volume_load_failed)
        thread.finished.connect(self._on_volume_thread_finished)
        self._volume_thread = thread
        thread.start()

    def _on_volume_loaded(self, arr, spacing, source_label, directory: Path):
        self.volume = _to_uint8(arr)
        self.volume_spacing = spacing

        for key, ax in (("ax", self.ax_ax), ("sag", self.ax_sag), ("cor", self.ax_cor)):
            art = self._plot_artists.pop(key, None)
            # Le titre (et l'annotation de spacing du panneau CORONAL) sont des
            # textes de FIGURE (pas d'axes, cf. create_view_canvas / update_plots) :
            # ax.clear() ne les supprime pas, il faut le faire ici explicitement,
            # sinon l'ancien texte se superpose au nouveau.
            if art:
                for fig_key in ('title', 'extra'):
                    artist = art.get(fig_key)
                    if artist is not None:
                        try:
                            artist.remove()
                        except Exception:
                            pass
            ax.clear(); ax.set_facecolor('black'); ax.axis('off')

        z, y, x = self.volume.shape
        for slider, n in ((self.slider_ax, z), (self.slider_cor, y), (self.slider_sag, x)):
            slider.setEnabled(True)
            slider.setRange(0, n - 1)
            slider.setValue(n // 2)

        self._current_volume_load_id = self._log_loaded_file(
            "volume", f"Volume : {source_label} ({z}×{y}×{x} px)", directory)
        self._update_status()
        self.update_plots()

    def _on_volume_load_failed(self, message: str):
        QMessageBox.critical(self, "Erreur de chargement", message)
        self._update_status()

    def _on_volume_thread_finished(self):
        self._volume_thread.deleteLater()
        self._volume_thread = None
        self._set_loading_state(False)

    def load_reference_path(self, path: Path):
        if self._reference_thread is not None:
            return  # un chargement de référence est déjà en cours

        self._set_loading_state(True)
        thread = _LoadThread(lambda: load_reference_image(path), self)
        thread.succeeded.connect(
            lambda arr, spacing: self._on_reference_loaded(arr, spacing, path))
        thread.failed.connect(self._on_reference_load_failed)
        thread.finished.connect(self._on_reference_thread_finished)
        self._reference_thread = thread
        thread.start()

    def _on_reference_loaded(self, arr, spacing, path: Path):
        self.reference = _to_uint8(arr)
        self.reference_spacing = spacing
        h, w = self.reference.shape
        self._current_reference_load_id = self._log_loaded_file(
            "reference", f"Référence : {path.name} ({h}×{w} px)", path.parent)
        self._update_status()
        self._render_reference()

    def _on_reference_load_failed(self, message: str):
        QMessageBox.critical(self, "Erreur de chargement", message)
        self._update_status()

    def _on_reference_thread_finished(self):
        self._reference_thread.deleteLater()
        self._reference_thread = None
        self._set_loading_state(False)

    def _set_loading_state(self, loading: bool):
        """Curseur d'attente + statut + désactivation des boutons de chargement
        tant qu'un thread (volume ou référence) tourne encore.

        EFFACER TOUT est aussi désactivé pendant un chargement : sinon un clic
        pendant que le thread tourne encore effacerait l'affichage juste avant
        que le résultat du chargement en cours ne le réécrive, ce qui ferait
        réapparaître ce qui vient d'être effacé."""
        any_active = loading or self._volume_thread is not None or self._reference_thread is not None
        self.btn_vol.setEnabled(not any_active)
        self.btn_ref.setEnabled(not any_active)
        self.btn_clear.setEnabled(not any_active)
        if loading:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.lbl_status.setStyleSheet(_STATUS_STYLE_LOADING)
            self.lbl_status.setText("Chargement du volume/séquence en cours…")
        elif not any_active:
            QApplication.restoreOverrideCursor()

    def _log_loaded_file(self, kind: str, label: str, directory: Path) -> int:
        """Ajoute une entrée dans le panneau « FICHIERS CHARGÉS » (nom + dossier).

        Historique uniquement : n'affecte jamais les fichiers sur disque.
        Retourne un identifiant unique, utilisé pour savoir plus tard si cette
        entrée précise est encore celle actuellement affichée (cf. clic droit
        « Retirer »).
        """
        self._load_counter += 1
        load_id = self._load_counter
        item = QListWidgetItem(f"{label}\n{directory}")
        item.setData(Qt.UserRole, {"kind": kind, "load_id": load_id, "directory": directory})
        self.list_history.addItem(item)
        self.list_history.scrollToBottom()
        return load_id

    def _update_status(self):
        self.lbl_status.setStyleSheet(_STATUS_STYLE_NORMAL)
        lines = []
        if self.volume is not None:
            z, y, x = self.volume.shape
            lines.append(f"Volume : {z}×{y}×{x} px, Pixel spacing : {_spacing_str(self.volume_spacing)}")
        else:
            lines.append("Volume : non chargé")
        if self.reference is not None:
            h, w = self.reference.shape
            lines.append(f"Référence : {h}×{w} px, Pixel spacing : {_spacing_str(self.reference_spacing)}")
        else:
            lines.append("Référence : non chargée")
        self.lbl_status.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------

    def schedule_update(self):
        """Coalesce les rafales de signaux slider (drag rapide) en un seul redraw."""
        if self._update_pending or self._closing:
            return
        self._update_pending = True
        QTimer.singleShot(0, self._do_update)

    def _do_update(self):
        self._update_pending = False
        if self._closing:
            return
        try:
            self.update_plots()
        except RuntimeError:
            pass  # fenêtre fermée avant l'exécution de la mise à jour différée

    def update_plots(self):
        if self.volume is None:
            return

        z = min(self.slider_ax.value(), self.volume.shape[0] - 1)
        y = min(self.slider_cor.value(), self.volume.shape[1] - 1)
        x = min(self.slider_sag.value(), self.volume.shape[2] - 1)

        self.lbl_ax.setText(f"AXIAL (Slice {z + 1} / {self.volume.shape[0]})")
        self.lbl_cor.setText(f"CORONAL (Slice {y + 1} / {self.volume.shape[1]})")
        self.lbl_sag.setText(f"SAGITTAL (Slice {x + 1} / {self.volume.shape[2]})")

        def draw_image(key, ax, canvas, data, title, color, lh, lv, ch, cv, extra_text=None):
            art = self._plot_artists.get(key)
            if art is None:
                im = ax.imshow(data, cmap='gray', aspect='equal', interpolation='nearest',
                                vmin=0, vmax=255)
                hline = ax.axhline(lh, color=ch, linewidth=0.8, alpha=0.6)
                vline = ax.axvline(lv, color=cv, linewidth=0.8, alpha=0.6)
                # Texte de FIGURE (pas d'axes) : avec aspect='equal', l'axe se
                # retrecit aux proportions de la coupe et se recentre dans le
                # panneau -- un transAxes suivrait ce recentrage et le titre
                # se retrouverait au milieu du panneau au lieu du coin. Un
                # texte de figure reste lui ancre au coin du panneau.
                title_artist = ax.figure.text(0.02, 0.98, title, color=color,
                                               fontweight='bold', fontsize=16, va='top')
                extra_artist = None
                if extra_text:
                    extra_artist = ax.figure.text(0.02, 0.02, extra_text, color=color,
                                                   fontsize=12, va='bottom')
                self._plot_artists[key] = {
                    'im': im, 'hline': hline, 'vline': vline,
                    'title': title_artist, 'extra': extra_artist,
                }
            else:
                art['im'].set_data(data)
                art['hline'].set_ydata([lh, lh])
                art['vline'].set_xdata([lv, lv])
            canvas.draw_idle()

        draw_image("ax", self.ax_ax, self.canvas_ax, self.volume[z, :, :],
                   "AXIAL", _COLOR_AXIAL, y, x, "blue", "green")
        draw_image("sag", self.ax_sag, self.canvas_sag, self.volume[:, :, x],
                   "SAGITTAL", _COLOR_SAGITTAL, z, y, "red", "blue")
        draw_image("cor", self.ax_cor, self.canvas_cor, self.volume[:, y, :],
                   "CORONAL", _COLOR_CORONAL, z, x, "red", "green",
                   extra_text=_spacing_str(self.volume_spacing) if self.volume_spacing else None)

    def _render_reference(self):
        self.ax_ref.clear()
        self.ax_ref.set_facecolor('black')
        self.ax_ref.axis('off')

        # Textes de FIGURE (pas d'axes) : ax.clear() ne les supprime pas (cf.
        # draw_image dans update_plots pour le detail), donc on les retire nous-
        # memes avant d'en recreer, sinon ancien et nouveau titre se superposent.
        for artist in self._ref_title_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._ref_title_artists = []

        if self.reference is not None:
            self.ax_ref.imshow(self.reference, cmap='gray', aspect='equal',
                                interpolation='nearest', vmin=0, vmax=255)
            fig = self.ax_ref.figure
            t_title = fig.text(0.02, 0.98, "RÉFÉRENCE SAGITTALE",
                                color=_COLOR_REFERENCE, fontweight='bold', fontsize=16, va='top')
            t_spacing = fig.text(0.02, 0.02, _spacing_str(self.reference_spacing),
                                  color=_COLOR_REFERENCE, fontsize=12, va='bottom')
            self._ref_title_artists = [t_title, t_spacing]
        else:
            self.ax_ref.text(0.5, 0.5, "Aucune image de référence\nchargée",
                              transform=self.ax_ref.transAxes, color="#555555",
                              ha='center', va='center', fontsize=16)
        self.canvas_ref.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def handle_scroll(self, event, view_type):
        step = 1 if event.angleDelta().y() > 0 else -1
        if view_type == "axial": self.slider_ax.setValue(self.slider_ax.value() + step)
        elif view_type == "coronal": self.slider_cor.setValue(self.slider_cor.value() + step)
        elif view_type == "sagittal": self.slider_sag.setValue(self.slider_sag.value() + step)
        event.accept()

    def reset_views(self):
        if self.volume is None:
            return
        self.slider_ax.setValue(self.volume.shape[0] // 2)
        self.slider_cor.setValue(self.volume.shape[1] // 2)
        self.slider_sag.setValue(self.volume.shape[2] // 2)

    def clear_all(self):
        """Efface le volume et l'image de référence chargés, et remet les
        panneaux/sliders à leur état initial (vide)."""
        self._clear_volume_display()
        self._clear_reference_display()

        # Vide uniquement l'historique affiché dans l'interface : les fichiers
        # DICOM d'origine sur le disque ne sont jamais touchés par ce bouton.
        self.list_history.clear()

        self._update_status()

    def _clear_volume_display(self):
        """Efface uniquement le volume (AXIAL/CORONAL/SAGITTAL), pas la référence."""
        self.volume = None
        self.volume_spacing = None
        self._current_volume_load_id = None

        for key, ax, canvas in (
            ("ax", self.ax_ax, self.canvas_ax),
            ("sag", self.ax_sag, self.canvas_sag),
            ("cor", self.ax_cor, self.canvas_cor),
        ):
            art = self._plot_artists.pop(key, None)
            if art:
                for fig_key in ('title', 'extra'):
                    artist = art.get(fig_key)
                    if artist is not None:
                        try:
                            artist.remove()
                        except Exception:
                            pass
            self._draw_placeholder(ax, "Aucun volume chargé")
            canvas.draw_idle()

        for slider, lbl, name in (
            (self.slider_ax, self.lbl_ax, "AXIAL"),
            (self.slider_cor, self.lbl_cor, "CORONAL"),
            (self.slider_sag, self.lbl_sag, "SAGITTAL"),
        ):
            slider.setEnabled(False)
            slider.setRange(0, 0)
            slider.setValue(0)
            lbl.setText(f"{name} (Slice 1 / 1)")

    def _clear_reference_display(self):
        """Efface uniquement l'image de référence, pas le volume."""
        self.reference = None
        self.reference_spacing = None
        self._current_reference_load_id = None
        self._render_reference()

    # ------------------------------------------------------------------
    # Panneau "FICHIERS CHARGÉS" : ouvrir le dossier / retirer une entrée
    # ------------------------------------------------------------------

    def _show_history_context_menu(self, pos):
        item = self.list_history.itemAt(pos)
        if item is None:
            return
        data = item.data(Qt.UserRole)
        menu = QMenu(self.list_history)
        menu.addAction("Ouvrir le dossier", lambda: self._open_folder(data["directory"]))
        menu.addAction("Retirer de la liste", lambda: self._remove_history_item(item))
        menu.exec(self.list_history.mapToGlobal(pos))

    def _open_folder(self, directory: Path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _remove_history_item(self, item: QListWidgetItem):
        """Retire une entrée de l'historique. Si c'est précisément l'entrée à
        l'origine de l'affichage courant (volume ou référence), cet affichage
        disparaît aussi ; une entrée déjà remplacée par un chargement plus
        récent ne fait, elle, que disparaître de la liste."""
        data = item.data(Qt.UserRole)
        row = self.list_history.row(item)
        self.list_history.takeItem(row)

        if data["kind"] == "volume" and data["load_id"] == self._current_volume_load_id:
            self._clear_volume_display()
        elif data["kind"] == "reference" and data["load_id"] == self._current_reference_load_id:
            self._clear_reference_display()
        self._update_status()

    def closeEvent(self, event):
        """Coupe toute source de repaint avant que les widgets ne disparaissent.

        Un singleShot en attente (schedule_update) ou un draw_idle() encore en
        file peut s'exécuter sur des widgets déjà détruits -> crash Qt natif
        (access violation, aucune exception Python). On désactive les callbacks
        avant de laisser la fermeture se poursuivre.
        """
        self._closing = True
        # Un _LoadThread encore actif emettrait succeeded/failed/finished sur
        # des widgets sur le point d'etre detruits. On le laisse terminer
        # (chargement deja bien avance a ce stade) plutot que de le tuer, puis
        # on deconnecte ses signaux : wait() bloque le thread GUI, donc aucun
        # signal en attente n'a encore pu etre livre (la boucle d'evenements
        # ne tourne pas pendant le wait) -- les deconnecter ici les rend
        # inoffensifs quand elle reprendra apres cette fonction.
        for thread in (self._volume_thread, self._reference_thread):
            if thread is not None:
                thread.wait()
                for sig in (thread.succeeded, thread.failed, thread.finished):
                    try:
                        sig.disconnect()
                    except (RuntimeError, TypeError):
                        pass
        for slider in (self.slider_ax, self.slider_cor, self.slider_sag):
            try:
                slider.valueChanged.disconnect(self.schedule_update)
            except (RuntimeError, TypeError):
                pass
        for canvas in (self.canvas_ax, self.canvas_sag, self.canvas_cor, self.canvas_ref):
            try:
                canvas.wheelEvent = lambda event: None
            except RuntimeError:
                pass
        super().closeEvent(event)


# ── Entry point ──────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualiseur DICOM multi-vues (AXIAL / CORONAL / SAGITTAL "
                    "+ référence sagittale).")
    parser.add_argument("volume", nargs="?", default=None,
                         help="Séquence dynamique / volume 3D DICOM : fichier "
                              "multi-trame ou dossier de coupes.")
    parser.add_argument("reference", nargs="?", default=None,
                         help="Image sagittale de référence DICOM (optionnelle).")
    return parser.parse_args()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(_resource_path("icon.ico"))))
    args = _parse_args()

    window = DicomViewerWindow()
    if args.volume:
        window.load_volume_path(Path(args.volume))
    if args.reference:
        window.load_reference_path(Path(args.reference))

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
