import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QSlider, QLabel, QMessageBox, QPushButton)
from PySide6.QtCore import (Qt, QCoreApplication, QEvent, QEventLoop, QTimer,
                            Signal)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors


class MITKViewer(QMainWindow):
    # Emis depuis closeEvent : permet a valider_recalage() d'attendre la
    # fermeture SANS avoir a detruire la fenetre (cf. WA_DeleteOnClose, qui
    # provoquait un crash natif Qt6Gui.dll — voir closeEvent ci-dessous).
    closed = Signal()

    def __init__(self, vol_mitk, mask_mitk=None):
        super().__init__()
        self.setWindowTitle("Visualiseur Médical - Multi-Vues (Volume + Seg)")
        self.setStyleSheet("background-color: #121212; color: white;")

        self.volume = vol_mitk
        self.mask = mask_mitk  # Stockage du masque (déjà transposé)

        # Création d'une colormap : 0 = transparent, 1 = Rouge vif
        # On utilise une liste de couleurs pour ListedColormap
        self.cmap_mask = mcolors.ListedColormap([(0, 0, 0, 0), (1, 0, 0, 1)])  # RGBA

        # --- LAYOUT PRINCIPAL ---
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- SIDEBAR ---
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(15, 20, 15, 20)

        title_nav = QLabel("NAVIGATION")
        title_nav.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #AAAAAA;")
        sidebar.addWidget(title_nav)

        # Initialisation basée sur les dimensions réelles du volume
        self.slider_ax, self.lbl_ax = self.create_mitk_slider("AXIAL", self.volume.shape[0], sidebar, "#FF5555")
        self.slider_cor, self.lbl_cor = self.create_mitk_slider("CORONAL", self.volume.shape[1], sidebar, "#5555FF")
        self.slider_sag, self.lbl_sag = self.create_mitk_slider("SAGITTAL", self.volume.shape[2], sidebar, "#55FF55")

        sidebar.addSpacing(20)

        self.btn_reset = QPushButton("RESET VIEWS")
        self.btn_reset.setMinimumHeight(45)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #00E5FF; color: #000000; font-weight: bold;
                border-radius: 8px; border: 2px solid #00B8D4;
            }
            QPushButton:hover { background-color: #64FFDA; }
        """)
        self.btn_reset.clicked.connect(self.reset_views)
        sidebar.addWidget(self.btn_reset)

        sidebar.addStretch()
        main_layout.addLayout(sidebar, 1)

        # --- GRILLE DES VUES ---
        grid_layout = QVBoxLayout()
        row1, row2 = QHBoxLayout(), QHBoxLayout()

        self.canvas_ax, self.ax_ax = self.create_view_canvas("axial")
        self.canvas_sag, self.ax_sag = self.create_view_canvas("sagittal")
        self.canvas_cor, self.ax_cor = self.create_view_canvas("coronal")
        self.canvas_empty, self.ax_empty = self.create_view_canvas("3d")

        row1.addWidget(self.canvas_ax); row1.addWidget(self.canvas_sag)
        row2.addWidget(self.canvas_cor); row2.addWidget(self.canvas_empty)

        grid_layout.addLayout(row1); grid_layout.addLayout(row2)
        main_layout.addLayout(grid_layout, 5)

        self._plot_artists = {}
        self._update_pending = False
        self._closing = False

        self.update_plots()
        self.showMaximized()

    def create_mitk_slider(self, name, max_val, layout, color):
        lbl = QLabel(f"{name} (Slice 1 / {max_val})")
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; margin-top: 15px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max_val - 1)
        slider.setValue(max_val // 2)
        slider.valueChanged.connect(self.schedule_update)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider, lbl

    def create_view_canvas(self, view_type):
        fig = Figure(facecolor='black')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.axis('off')
        canvas.wheelEvent = lambda event: self.handle_scroll(event, view_type)
        return canvas, ax

    def schedule_update(self):
        """Coalesce bursts of slider signals (fast drag) into one redraw per tick."""
        if self._update_pending or self._closing:
            return
        self._update_pending = True
        QTimer.singleShot(0, self._do_update)

    def _do_update(self):
        self._update_pending = False
        if self._closing:
            return  # fermeture en cours : les widgets ne sont plus fiables
        try:
            self.update_plots()
        except RuntimeError:
            pass  # window was closed before the deferred update ran

    def closeEvent(self, event):
        """Coupe toute source de repaint AVANT que les widgets ne disparaissent.

        Un singleShot en attente (schedule_update) ou un draw_idle() encore en
        file s'executait sur des widgets deja detruits -> access violation
        0xc0000005 dans Qt6Gui.dll, process tue net (aucune exception Python).
        On deconnecte les sliders et on neutralise les callbacks avant de
        laisser la fermeture se poursuivre.
        """
        self._closing = True
        for slider in (self.slider_ax, self.slider_cor, self.slider_sag):
            try:
                slider.valueChanged.disconnect(self.schedule_update)
            except (RuntimeError, TypeError):
                pass
        for canvas in (self.canvas_ax, self.canvas_sag,
                       self.canvas_cor, self.canvas_empty):
            try:
                canvas.wheelEvent = lambda event: None
            except RuntimeError:
                pass
        self.closed.emit()
        super().closeEvent(event)

    def handle_scroll(self, event, view_type):
        step = 1 if event.angleDelta().y() > 0 else -1
        if view_type == "axial": self.slider_ax.setValue(self.slider_ax.value() + step)
        elif view_type == "coronal": self.slider_cor.setValue(self.slider_cor.value() + step)
        elif view_type == "sagittal": self.slider_sag.setValue(self.slider_sag.value() + step)
        event.accept()

    def reset_views(self):
        self.slider_ax.setValue(self.volume.shape[0] // 2)
        self.slider_cor.setValue(self.volume.shape[1] // 2)
        self.slider_sag.setValue(self.volume.shape[2] // 2)

    def update_plots(self):
        z, y, x = self.slider_ax.value(), self.slider_cor.value(), self.slider_sag.value()

        # Sécurité pour ne pas sortir des dimensions après resize potentiel
        z = min(z, self.volume.shape[0] - 1)
        y = min(y, self.volume.shape[1] - 1)
        x = min(x, self.volume.shape[2] - 1)

        self.lbl_ax.setText(f"AXIAL (Slice {z + 1} / {self.volume.shape[0]})")
        self.lbl_cor.setText(f"CORONAL (Slice {y + 1} / {self.volume.shape[1]})")
        self.lbl_sag.setText(f"SAGITTAL (Slice {x + 1} / {self.volume.shape[2]})")

        def draw_image(key, ax, canvas, data, mask_slice, title, color, lh, lv, ch, cv):
            art = self._plot_artists.get(key)
            has_mask = mask_slice is not None and mask_slice.shape == data.shape
            m_bin = (mask_slice > 0).astype(np.uint8) if has_mask else None

            if art is None:
                # Premier affichage de cette vue : on crée les artistes une fois.
                im = ax.imshow(data, cmap='gray', aspect='equal', interpolation='nearest',
                                vmin=0, vmax=255)
                mask_im = None
                if has_mask:
                    mask_im = ax.imshow(m_bin, cmap=self.cmap_mask, alpha=0.5, aspect='equal',
                                         interpolation='nearest', zorder=10, vmin=0, vmax=1)
                hline = ax.axhline(lh, color=ch, linewidth=0.8, alpha=0.6)
                vline = ax.axvline(lv, color=cv, linewidth=0.8, alpha=0.6)
                ax.text(0.02, 0.98, title, transform=ax.transAxes, color=color, fontweight='bold', va='top')
                self._plot_artists[key] = {'im': im, 'mask_im': mask_im, 'hline': hline, 'vline': vline}
            else:
                # Appels suivants : on met juste les données à jour (pas de clear/recreate).
                art['im'].set_data(data)
                if art['mask_im'] is not None and has_mask:
                    art['mask_im'].set_data(m_bin)
                art['hline'].set_ydata([lh, lh])
                art['vline'].set_xdata([lv, lv])

            canvas.draw_idle()

        # Tranches Volume et Masque
        draw_image("ax", self.ax_ax, self.canvas_ax, self.volume[z, :, :],
                   self.mask[z, :, :] if self.mask is not None else None,
                   "AXIAL", "#FF5555", y, x, "blue", "green")

        draw_image("sag", self.ax_sag, self.canvas_sag, self.volume[:, :, x],
                   self.mask[:, :, x] if self.mask is not None else None,
                   "SAGITTAL", "#55FF55", z, y, "red", "blue")

        draw_image("cor", self.ax_cor, self.canvas_cor, self.volume[:, y, :],
                   self.mask[:, y, :] if self.mask is not None else None,
                   "CORONAL", "#5555FF", z, x, "red", "green")

        self.ax_empty.axis('off')
        self.canvas_empty.draw_idle()


# ── Exceptions ───────────────────────────────────────────────────────────────

class ValidationRefusee(Exception):
    """Levée quand l'utilisateur clique 'Non' dans la boîte de validation."""


# ── Entry point ──────────────────────────────────────────────────────────────

def valider_recalage(volume_data, mask_data=None):
    # --- GESTION DES DIMENSIONS ---
    if mask_data is not None and volume_data.shape != mask_data.shape:
        print(f"Mismatch: Volume {volume_data.shape} vs Mask {mask_data.shape}. Adaptation...")
        new_mask = np.zeros(volume_data.shape, dtype=mask_data.dtype)
        z_c, y_c, x_c = [min(v, m) for v, m in zip(volume_data.shape, mask_data.shape)]
        new_mask[:z_c, :y_c, :x_c] = mask_data[:z_c, :y_c, :x_c]
        mask_data = new_mask

    # --- TRANSPOSITION ET NORMALISATION ---
    vol_mitk = np.transpose(volume_data, (0, 2, 1))
    max_v = np.max(vol_mitk) if np.max(vol_mitk) > 0 else 1.0
    data_ui = (vol_mitk / max_v * 255).astype(np.uint8)

    mask_ui = None
    if mask_data is not None:
        mask_ui = np.transpose(mask_data, (0, 2, 1))

    app = QApplication.instance() or QApplication(sys.argv)

    # Qt fait qFatal (= tue le process, increcevable) si on cree une fenetre
    # alors qu'aucun ecran n'est disponible. Cela arrive pendant une
    # reconfiguration d'affichage -- un moniteur DisplayPort qui s'endort se
    # deconnecte. Vu en production : l'etape 4 a ouvert ce viewer a la seconde
    # ou un ecran disparaissait -> "Cannot create window: no screens
    # available", travail perdu alors qu'il etait termine.
    from tools.ui_utils import attendre_ecran_disponible, bring_to_front
    if not attendre_ecran_disponible():
        raise RuntimeError(
            "Aucun écran disponible : impossible d'ouvrir le visualiseur de "
            "validation. Vérifiez que l'écran n'est pas en veille, puis "
            "relancez cette étape.")

    viewer = MITKViewer(data_ui, mask_ui)
    viewer.show()
    # Le viewer bloque le pipeline jusqu'a sa fermeture : ouvert derriere la
    # fenetre principale (maximisee), l'app parait figee. show() seul ne suffit
    # pas -- Windows ignore le focus demande par un process d'arriere-plan.
    bring_to_front(viewer)

    # NE PAS utiliser WA_DeleteOnClose ici : la fenetre etait detruite des la
    # fermeture, alors que des repaints (draw_idle) et un eventuel singleShot
    # etaient encore en file -> ils s'executaient sur des widgets liberes et
    # tuaient le process (access violation Qt6Gui.dll), typiquement au passage
    # repere commun -> segmentation. On attend le signal 'closed', on vide la
    # file d'evenements pendant que les widgets sont encore vivants, et on ne
    # detruit qu'ensuite.
    loop = QEventLoop()
    viewer.closed.connect(loop.quit)
    viewer.destroyed.connect(loop.quit)  # filet de securite : jamais de blocage
    loop.exec()

    # --- Teardown : c'est QT qui doit detruire la fenetre, pas Python. ---
    #
    # L'ancienne sequence (deleteLater + processEvents + 'viewer = None')
    # reposait sur deux croyances fausses, verifiees par test :
    #
    #  - processEvents() n'execute PAS les DeferredDelete. Qt les reserve a la
    #    boucle d'evenements qui etait active lors du deleteLater(), pour ne
    #    pas detruire d'objets sous une boucle imbriquee. La destruction
    #    n'avait donc jamais lieu la, contrairement a ce qu'affirmait le
    #    commentaire d'origine.
    #  - 'viewer = None' lachait la DERNIERE reference Python. La fenetre
    #    n'ayant pas de parent, elle appartient a Python : shiboken detruisait
    #    donc l'objet C++ sur-le-champ, hors du controle de Qt.
    #
    # La fenetre disparaissait ainsi sans que Qt ne la desenregistre, et la
    # liste d'ecrans de QGuiApplication gardait une entree morte. Au premier
    # repaint suivant :
    #   QWidget::metric -> QWidget::screen -> QWidgetPrivate::associatedScreen
    #   -> QGuiApplication::screenAt -> QScreen::virtualSiblings -> boom
    # (access violation Qt6Gui.dll, aucune exception Python), typiquement au
    # passage repere commun -> segmentation.
    viewer.hide()
    # Purge les repaints en attente pendant que les widgets sont VIVANTS.
    # ExcludeUserInputEvents : un clic distribue ici relancerait du dessin.
    app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    # Plus rien ne doit referencer le viewer au moment ou il disparait.
    for _sig in (viewer.closed, viewer.destroyed):
        try:
            _sig.disconnect()
        except (RuntimeError, TypeError):
            pass
    viewer.deleteLater()
    # LE point qui manquait : forcer l'execution du DeferredDelete. C'est Qt
    # qui detruit alors la fenetre, et lui seul purge les evenements postes et
    # ses propres references internes.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    viewer = None

    from tools.ui_utils import exec_msgbox
    _mb = QMessageBox()
    _mb.setWindowTitle("Decision")
    _mb.setText("Passer a l'etape suivante ?")
    _mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    _mb.setDefaultButton(QMessageBox.No)
    _mb.setIcon(QMessageBox.Question)
    # Juste après la fermeture du viewer MITK, Windows peut renvoyer le focus
    # à la fenêtre principale au lieu de cette boîte : sans ceci elle s'ouvre
    # masquée et la fenêtre principale semble figée indéfiniment (elle attend
    # en réalité un clic Oui/Non sur une boîte invisible).
    _mb.setWindowFlags(_mb.windowFlags() | Qt.WindowStaysOnTopHint)
    _mb.raise_()
    _mb.activateWindow()
    if exec_msgbox(_mb) != QMessageBox.Yes:
        raise ValidationRefusee("Etape refusee par l'utilisateur.")


if __name__ == "__main__":
    # Test avec un volume 320x320 et un masque 388 (pour simuler votre erreur)
    data_test = np.random.randint(0, 100, (320, 256, 256))
    mask_test = np.zeros((388, 256, 256))  # Plus grand que le volume
    mask_test[100:200, 100:200, 100:200] = 1

    valider_recalage(data_test, mask_test)
