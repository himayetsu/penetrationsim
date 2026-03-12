# -*- coding: utf-8 -*-
"""Main window: docks, scene tree, properties, simulation controls, visuals."""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QScrollArea, QDockWidget, QCheckBox,
    QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QLinearGradient, QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from physics import PhysicsEngine
from bodies import Penetrator, ArmorPlate
from views import SimViewWidget, OrthoCanvas
from gizmo import TransformGizmo
from materials import MATERIALS

# Name (display) -> material key, for change_mat
NAME_TO_MATERIAL = {m['name']: key for key, m in MATERIALS.items()}

# Shared velocity colormap used by both 3D scatter and the legend widget.
# (t=0 → slow/blue, t=1 → fast/red)
VEL_MAX = 1800.0
VEL_STOPS_T = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00], dtype=np.float32)
VEL_STOPS_RGB = np.array([
    [0.10, 0.20, 1.00],  # dark blue  (0 m/s)
    [0.05, 0.80, 1.00],  # cyan       (360 m/s)
    [0.10, 1.00, 0.40],  # green      (720 m/s)
    [1.00, 1.00, 0.10],  # yellow     (1080 m/s)
    [1.00, 0.45, 0.05],  # orange     (1440 m/s)
    [1.00, 0.05, 0.05],  # red        (1800 m/s)
], dtype=np.float32)


class VelocityLegend(QWidget):
    """Compact velocity colorbar overlaid on the 3D viewport."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(82, 210)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        BAR_X, BAR_Y, BAR_W, BAR_H = 10, 20, 14, 160
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Dark background panel
        p.fillRect(0, 0, self.width(), self.height(), QColor(18, 18, 18, 190))

        # Gradient bar: bottom = slow (blue), top = fast (red)
        grad = QLinearGradient(BAR_X, BAR_Y + BAR_H, BAR_X, BAR_Y)
        for t, rgb in zip(VEL_STOPS_T, VEL_STOPS_RGB):
            grad.setColorAt(float(t), QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
        p.fillRect(BAR_X, BAR_Y, BAR_W, BAR_H, grad)
        p.setPen(QColor(55, 55, 55))
        p.drawRect(BAR_X, BAR_Y, BAR_W - 1, BAR_H - 1)

        # Tick marks and labels
        p.setFont(QFont('Segoe UI', 7))
        for v in (0, 360, 720, 1080, 1440, 1800):
            t = v / VEL_MAX
            y = int(BAR_Y + BAR_H * (1.0 - t))
            p.setPen(QColor(130, 130, 130))
            p.drawLine(BAR_X + BAR_W, y, BAR_X + BAR_W + 4, y)
            p.setPen(QColor(210, 210, 210))
            p.drawText(BAR_X + BAR_W + 6, y + 4, str(v))

        # "m/s" rotated title
        p.setPen(QColor(170, 170, 170))
        p.setFont(QFont('Segoe UI', 7))
        p.save()
        p.translate(BAR_X - 3, BAR_Y + BAR_H // 2 + 14)
        p.rotate(-90)
        p.drawText(-14, 0, 'm/s')
        p.restore()
        p.end()


def _materials_for_role(role):
    """Return list of display names for materials usable as penetrator or armor."""
    return [m['name'] for m in MATERIALS.values() if m.get('role') == role or m.get('role') == 'both']


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArmorSim 3D - Soft Body Simulation")
        self.setGeometry(50, 50, 1500, 950)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1a1a1a; color: #ccc; font-family: 'Segoe UI'; }
            QDockWidget { background: #222; border: 1px solid #333; }
            QDockWidget::title { background: #2a2a2a; padding: 5px; }
            QGroupBox { border: 1px solid #3a3a3a; margin-top: 8px; padding-top: 8px; background: #252525; }
            QGroupBox::title { color: #6af; }
            QPushButton { background: #333; border: 1px solid #444; padding: 5px 10px; }
            QPushButton:hover { background: #444; }
            QPushButton:checked { background: #4680c2; }
            QSlider::groove:horizontal { height: 4px; background: #333; }
            QSlider::handle:horizontal { background: #6af; width: 12px; margin: -4px 0; border-radius: 6px; }
            QComboBox, QSpinBox { background: #333; border: 1px solid #444; padding: 3px; }
            QTreeWidget { background: #222; border: 1px solid #333; }
            QTreeWidget::item:selected { background: #4680c2; }
        """)

        self.physics = PhysicsEngine()
        self.running = False
        self.simulation_started = False
        self.selected = None
        self.show_springs = True
        self.quad_mode = False
        self.ortho_views = []
        self.show_penetrator = True
        self.show_armor = True

        self.setup_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.create_scene()

    def setup_ui(self):
        self.view_container = QWidget()
        self.view_grid = QGridLayout(self.view_container)
        self.view_grid.setContentsMargins(0, 0, 0, 0)
        self.view_grid.setSpacing(2)

        self.view = SimViewWidget()
        self.view.main_window = self
        self.view.setBackgroundColor(pg.mkColor(22, 24, 28))
        self.gizmo = TransformGizmo(self.view)
        self.view.gizmo = self.gizmo

        grid_item = gl.GLGridItem()
        grid_item.setSize(0.8, 0.8, 1)
        grid_item.setSpacing(0.04, 0.04)
        grid_item.setColor((35, 35, 35, 80))
        self.view.addItem(grid_item)

        self.view_grid.addWidget(self.view, 0, 0, 2, 2)
        self.setCentralWidget(self.view_container)

        self.vel_legend = VelocityLegend(self.view)
        self.vel_legend.show()

        scene_dock = QDockWidget("Scene", self)
        scene_w = QWidget()
        scene_l = QVBoxLayout(scene_w)
        scene_l.setContentsMargins(4, 4, 4, 4)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Objects"])
        self.tree.itemClicked.connect(self.on_select)
        scene_l.addWidget(self.tree)

        btn_row = QHBoxLayout()
        btn_pen = QPushButton("+ Penetrator")
        btn_pen.clicked.connect(self.add_penetrator)
        btn_armor = QPushButton("+ Armor")
        btn_armor.clicked.connect(self.add_armor)
        btn_row.addWidget(btn_pen)
        btn_row.addWidget(btn_armor)
        scene_l.addLayout(btn_row)

        btn_del = QPushButton("Delete")
        btn_del.clicked.connect(self.delete_selected)
        scene_l.addWidget(btn_del)

        scene_dock.setWidget(scene_w)
        self.addDockWidget(Qt.LeftDockWidgetArea, scene_dock)

        props_dock = QDockWidget("Properties", self)
        props_dock.setMinimumWidth(250)
        props_scroll = QScrollArea()
        props_scroll.setWidgetResizable(True)
        props_w = QWidget()
        self.props_layout = QVBoxLayout(props_w)
        self.props_layout.setContentsMargins(4, 4, 4, 4)
        self.props_container = QWidget()
        self.props_box = QVBoxLayout(self.props_container)
        self.props_layout.addWidget(self.props_container)
        self.props_layout.addStretch()
        props_scroll.setWidget(props_w)
        props_dock.setWidget(props_scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, props_dock)

        tl_dock = QDockWidget("Simulation", self)
        tl_dock.setMaximumHeight(130)
        tl_w = QWidget()
        tl_l = QVBoxLayout(tl_w)
        tl_l.setContentsMargins(6, 4, 6, 4)

        row1 = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)
        row1.addWidget(self.play_btn)

        reset_btn = QPushButton("⏮ Reset")
        reset_btn.clicked.connect(self.reset_sim)
        row1.addWidget(reset_btn)

        row1.addSpacing(10)
        row1.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(2)
        self.speed_slider.valueChanged.connect(self.on_speed)
        row1.addWidget(self.speed_slider)
        self.speed_lbl = QLabel("0.002x")
        self.speed_lbl.setMinimumWidth(50)
        row1.addWidget(self.speed_lbl)

        row1.addSpacing(10)
        self.spring_cb = QCheckBox("Show Bonds")
        self.spring_cb.setChecked(True)
        self.spring_cb.toggled.connect(self.toggle_springs)
        row1.addWidget(self.spring_cb)

        row1.addStretch()
        self.time_lbl = QLabel("Time: 0.0 μs")
        self.time_lbl.setStyleSheet("color: #6af; font-weight: bold;")
        row1.addWidget(self.time_lbl)
        self.sim_speed_lbl = QLabel("Sim Speed: 0.002x")
        row1.addWidget(self.sim_speed_lbl)
        self.pen_lbl = QLabel("Pen: 0 mm")
        row1.addWidget(self.pen_lbl)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("View:"))
        self.quad_btn = QPushButton("Quad View")
        self.quad_btn.setCheckable(True)
        self.quad_btn.clicked.connect(self.toggle_quad_view)
        row2.addWidget(self.quad_btn)

        self.ortho_btn = QPushButton("Ortho")
        self.ortho_btn.setCheckable(True)
        self.ortho_btn.clicked.connect(self.toggle_ortho)
        row2.addWidget(self.ortho_btn)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Perspective", "Front", "Right", "Top"])
        self.view_combo.currentTextChanged.connect(self.set_view_preset)
        self.view_combo.setMinimumWidth(110)
        row2.addWidget(self.view_combo)

        row2.addSpacing(10)
        self.pen_vis_cb = QCheckBox("Projectile")
        self.pen_vis_cb.setChecked(True)
        self.pen_vis_cb.toggled.connect(self.toggle_penetrator_vis)
        row2.addWidget(self.pen_vis_cb)

        self.armor_vis_cb = QCheckBox("Armor")
        self.armor_vis_cb.setChecked(True)
        self.armor_vis_cb.toggled.connect(self.toggle_armor_vis)
        row2.addWidget(self.armor_vis_cb)

        row2.addStretch()

        tl_l.addLayout(row1)
        tl_l.addLayout(row2)
        tl_dock.setWidget(tl_w)
        self.addDockWidget(Qt.BottomDockWidgetArea, tl_dock)

        self._armor_rebuild_timer = QTimer(self)
        self._armor_rebuild_timer.setSingleShot(True)
        self._armor_rebuild_body = None
        self._armor_rebuild_timer.timeout.connect(self._do_armor_rebuild)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'vel_legend'):
            lw, lh = self.vel_legend.width(), self.vel_legend.height()
            self.vel_legend.move(self.view.width() - lw - 12,
                                 self.view.height() - lh - 12)

    def toggle_ortho(self):
        self.view.ortho = self.ortho_btn.isChecked()
        lbl = 'Ortho' if self.view.ortho else 'Perspective'
        self.view.view_mode = 'perspective' if not self.view.ortho else self.view.view_mode
        self.view._view_label.setText(lbl)
        self.view_combo.blockSignals(True)
        self.view_combo.setCurrentText('Perspective' if not self.view.ortho else 'Front')
        self.view_combo.blockSignals(False)
        self.view.repaint()

    def set_view_preset(self, name):
        presets = {
            'Perspective': {'ortho': False, 'elevation': 15, 'azimuth': 90, 'mode': 'perspective'},
            'Front':       {'ortho': True,  'elevation': 0,  'azimuth': 90, 'mode': 'front'},
            'Right':       {'ortho': True,  'elevation': 0,  'azimuth': 0,  'mode': 'right'},
            'Top':         {'ortho': True,  'elevation': 90, 'azimuth': 90, 'mode': 'top'},
        }
        p = presets.get(name)
        if not p:
            return
        self.view.ortho = p['ortho']
        self.view.view_mode = p['mode']
        self.view.setCameraPosition(elevation=p['elevation'], azimuth=p['azimuth'])
        self.ortho_btn.blockSignals(True)
        self.ortho_btn.setChecked(p['ortho'])
        self.ortho_btn.blockSignals(False)
        self.view._view_label.setText(name)
        self.view.repaint()

    def toggle_quad_view(self):
        if self.quad_mode:
            for v in self.ortho_views:
                self.view_grid.removeWidget(v)
                v.deleteLater()
            self.ortho_views = []
            self.view_grid.addWidget(self.view, 0, 0, 2, 2)
            self.quad_mode = False
        else:
            self.view_grid.removeWidget(self.view)
            self.view_grid.addWidget(self.view, 1, 1)

            for mode, row, col in [('top', 0, 0), ('front', 0, 1), ('right', 1, 0)]:
                ov = OrthoCanvas(mode=mode)
                ov.show_springs = self.show_springs
                ov.set_bodies(self.physics.bodies)
                self.view_grid.addWidget(ov, row, col)
                self.ortho_views.append(ov)

            self.quad_mode = True

    def _sync_ortho_views(self):
        for ov in self.ortho_views:
            ov.set_bodies(self.physics.bodies)
            ov.show_springs = self.show_springs

    def _do_armor_rebuild(self):
        if self._armor_rebuild_body is not None:
            body = self._armor_rebuild_body
            self._armor_rebuild_body = None
            body.rebuild()
            self.rebuild_visuals(body)

    def create_scene(self):
        self.add_penetrator()
        self.add_armor()
        self.update_tree()
        if self.physics.bodies:
            self.select_body(self.physics.bodies[0])

    def add_penetrator(self):
        n = len([b for b in self.physics.bodies if b.body_type == 'penetrator'])
        p = Penetrator(f"Penetrator.{n + 1:03d}")
        self.physics.bodies.append(p)
        self.create_visuals(p)
        self.update_tree()
        self.select_body(p)

    def add_armor(self):
        a = ArmorPlate(f"Armor.{len([b for b in self.physics.bodies if b.body_type == 'armor']) + 1:03d}")
        self.physics.bodies.append(a)
        self.create_visuals(a)
        self.update_tree()
        self.select_body(a)

    def create_visuals(self, body):
        mat = MATERIALS[body.material]
        colors = np.zeros((body.n_particles, 4), dtype=np.float32)
        colors[:, :3] = mat['color']
        colors[:, 3] = 1.0

        scatter = gl.GLScatterPlotItem(pos=body.pos, color=colors, size=6, pxMode=True)
        scatter.setGLOptions('opaque')
        body.scatter = scatter
        self.view.addItem(scatter)

        if body.springs is not None and len(body.springs) > 0:
            lp = np.zeros((len(body.springs) * 2, 3), dtype=np.float32)
            for k, (i, j) in enumerate(body.springs):
                lp[k * 2] = body.pos[i]
                lp[k * 2 + 1] = body.pos[j]
            lines = gl.GLLinePlotItem(pos=lp, color=(*mat['color'] * 0.6, 0.3), width=1, mode='lines')
            body.lines = lines
            self.view.addItem(lines)

    def toggle_penetrator_vis(self, show):
        self.show_penetrator = show
        for body in self.physics.bodies:
            if body.body_type == 'penetrator':
                self.update_visuals(body)

    def toggle_armor_vis(self, show):
        self.show_armor = show
        for body in self.physics.bodies:
            if body.body_type == 'armor':
                self.update_visuals(body)

    def update_visuals(self, body):
        if body.scatter is None:
            return

        # Visibility toggle
        visible = (self.show_penetrator if body.body_type == 'penetrator'
                   else self.show_armor)
        body.scatter.setVisible(visible)
        if body.lines is not None:
            body.lines.setVisible(visible and self.show_springs
                                  and body.springs is not None
                                  and len(body.springs) > 0)

        colors = np.zeros((body.n_particles, 4), dtype=np.float32)
        if body.body_type == 'penetrator':
            # Velocity gradient on projectile
            vel_mag = np.sqrt(np.sum(body.vel ** 2, axis=1))
            vel_norm = np.clip(vel_mag / VEL_MAX, 0.0, 1.0)
            colors[:, 0] = np.interp(vel_norm, VEL_STOPS_T, VEL_STOPS_RGB[:, 0])
            colors[:, 1] = np.interp(vel_norm, VEL_STOPS_T, VEL_STOPS_RGB[:, 1])
            colors[:, 2] = np.interp(vel_norm, VEL_STOPS_T, VEL_STOPS_RGB[:, 2])
        else:
            # Armor uses flat material color
            colors[:, :3] = MATERIALS[body.material]['color']
        colors[:, 3] = 1.0
        if body.selected:
            colors[:, :3] = np.clip(colors[:, :3] + 0.1, 0, 1)
        body.scatter.setData(pos=body.pos, color=colors)

        if body.lines is not None and visible and self.show_springs:
            if body.springs is not None and len(body.springs) > 0:
                lp = np.zeros((len(body.springs) * 2, 3), dtype=np.float32)
                for k, (i, j) in enumerate(body.springs):
                    lp[k * 2] = body.pos[i]
                    lp[k * 2 + 1] = body.pos[j]
                body.lines.setData(pos=lp)

    def rebuild_visuals(self, body):
        if body.scatter:
            self.view.removeItem(body.scatter)
        if body.lines:
            self.view.removeItem(body.lines)
        body.scatter = None
        body.lines = None
        self.create_visuals(body)
        self.update_tree()

    def toggle_springs(self, show):
        self.show_springs = show
        for body in self.physics.bodies:
            if body.lines:
                body.lines.setVisible(show)
        for ov in self.ortho_views:
            ov.show_springs = show
            ov.update()

    def update_tree(self):
        self.tree.clear()
        root = QTreeWidgetItem(["Scene"])
        self.tree.addTopLevelItem(root)
        for i, body in enumerate(self.physics.bodies):
            springs = len(body.springs) if body.springs is not None else 0
            item = QTreeWidgetItem([f"{body.name} ({body.n_particles}p, {springs}s)"])
            item.setData(0, Qt.UserRole, i)
            c = QColor(255, 180, 80) if body.body_type == 'penetrator' else QColor(120, 160, 220)
            item.setForeground(0, c)
            root.addChild(item)
        root.setExpanded(True)

    def on_select(self, item, col):
        idx = item.data(0, Qt.UserRole)
        if idx is not None and 0 <= idx < len(self.physics.bodies):
            self.select_body(self.physics.bodies[idx])

    def select_body(self, body):
        if self.selected:
            self.selected.selected = False
            self.update_visuals(self.selected)
        self.selected = body
        if body:
            body.selected = True
            self.update_visuals(body)
            self.show_props(body)
            self.gizmo.attach(body)
        else:
            self.clear_props()
            self.gizmo.detach()

    def show_props(self, body):
        self.clear_props()
        info = QGroupBox("Info")
        il = QVBoxLayout(info)
        il.addWidget(QLabel(f"Name: {body.name}"))
        il.addWidget(QLabel(f"Particles: {body.n_particles}"))
        il.addWidget(QLabel(f"Springs: {len(body.springs) if body.springs is not None else 0}"))
        self.props_box.addWidget(info)

        mat_grp = QGroupBox("Material")
        ml = QVBoxLayout(mat_grp)
        mat_cb = QComboBox()
        role = 'penetrator' if body.body_type == 'penetrator' else 'armor'
        mat_cb.blockSignals(True)
        mat_cb.addItems(_materials_for_role(role))
        mat_cb.setCurrentText(MATERIALS[body.material]['name'])
        mat_cb.blockSignals(False)
        mat_cb.currentTextChanged.connect(lambda t: self.change_mat(body, t))
        ml.addWidget(mat_cb)
        self.props_box.addWidget(mat_grp)

        dim = QGroupBox("Dimensions")
        dl = QVBoxLayout(dim)
        if body.body_type == 'penetrator':
            self._spin("Length (mm)", 1, 10000, round(body.length * 1000),
                       lambda v: self.change_dim(body, 'length', v / 1000), dl)
            self._spin("Diameter (mm)", 1, 1000, round(body.diameter * 1000),
                       lambda v: self.change_dim(body, 'diameter', v / 1000), dl)
            self._spin("Velocity (m/s)", 1, 10000, round(body.initial_velocity),
                       lambda v: setattr(body, 'initial_velocity', float(v)), dl, 50)
        else:
            self._spin("Width (mm)", 1, 10000, round(body.width * 1000),
                       lambda v: self.change_dim(body, 'width', v / 1000), dl)
            self._spin("Height (mm)", 1, 10000, round(body.height * 1000),
                       lambda v: self.change_dim(body, 'height', v / 1000), dl)
            self._spin("Thickness (mm)", 1, 10000, round(body.thickness * 1000),
                       lambda v: self.change_dim(body, 'thickness', v / 1000), dl)
            self._spin("Angle (°)", 0, 90, body.angle,
                       lambda v: self.change_dim(body, 'angle', v), dl)
        self.props_box.addWidget(dim)

    def _spin(self, label, mn, mx, val, cb, layout, step=1):
        h = QHBoxLayout()
        h.addWidget(QLabel(label))
        s = QSpinBox()
        s.blockSignals(True)
        s.setRange(mn, mx)
        s.setValue(val)
        s.setSingleStep(step)
        s.blockSignals(False)
        s.valueChanged.connect(cb)
        h.addWidget(s)
        layout.addLayout(h)

    def clear_props(self):
        while self.props_box.count():
            w = self.props_box.takeAt(0).widget()
            if w:
                w.deleteLater()

    def change_mat(self, body, name):
        body.material = NAME_TO_MATERIAL.get(name, 'steel')
        body.rebuild()
        self.rebuild_visuals(body)

    def change_dim(self, body, attr, val):
        setattr(body, attr, val)
        if body.body_type == 'armor':
            self._armor_rebuild_body = body
            self._armor_rebuild_timer.stop()
            self._armor_rebuild_timer.start(350)
        else:
            body.rebuild()
            self.rebuild_visuals(body)

    def delete_selected(self):
        if self.selected:
            if self.selected.scatter:
                self.view.removeItem(self.selected.scatter)
            if self.selected.lines:
                self.view.removeItem(self.selected.lines)
            self.physics.bodies.remove(self.selected)
            self.selected = None
            self.gizmo.detach()
            self.update_tree()
            self.clear_props()

    def _save_reset_snapshot(self):
        """Save a full snapshot of all body arrays so reset_sim can restore them exactly."""
        self.physics.reset()  # put bodies in clean initial state first
        for b in self.physics.bodies:
            b._snap_rest_pos   = b.rest_pos.copy()
            b._snap_pos        = b.pos.copy()
            b._snap_vel        = b.vel.copy()
            b._snap_mass       = b.mass.copy()
            b._snap_active     = b.active.copy() if b.active is not None else None
            b._snap_radius     = b.radius.copy() if b.radius is not None else None
            b._snap_springs    = b.springs.copy() if b.springs is not None else None
            b._snap_spring_rest  = b.spring_rest.copy() if b.spring_rest is not None else None
            b._snap_spring_stiff = b.spring_stiff.copy() if b.spring_stiff is not None else None

    def toggle_play(self):
        self.running = self.play_btn.isChecked()
        if self.running:
            self.play_btn.setText("⏸ Pause")
            self._armor_rebuild_timer.stop()
            self._armor_rebuild_body = None
            if not self.simulation_started:
                self._save_reset_snapshot()
                for b in self.physics.bodies:
                    self.update_visuals(b)
                self.simulation_started = True
            self.gizmo.detach()
            self.timer.start(16)
        else:
            self.play_btn.setText("▶ Play")
            self.timer.stop()
            if self.selected:
                self.gizmo.attach(self.selected)

    def reset_sim(self):
        self.running = False
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶ Play")
        self.timer.stop()
        self._armor_rebuild_timer.stop()
        self._armor_rebuild_body = None
        for b in self.physics.bodies:
            if not hasattr(b, '_snap_rest_pos'):
                continue
            b.rest_pos    = b._snap_rest_pos.copy()
            b.pos         = b._snap_pos.copy()
            b.vel         = b._snap_vel.copy()
            b.mass        = b._snap_mass.copy()
            if b._snap_active is not None:
                b.active  = b._snap_active.copy()
            if b._snap_radius is not None:
                b.radius  = b._snap_radius.copy()
            if b._snap_springs is not None:
                b.springs      = b._snap_springs.copy()
                b.spring_rest  = b._snap_spring_rest.copy()
                b.spring_stiff = b._snap_spring_stiff.copy()
        # reset physics-engine counters and angular velocity only
        self.physics.time = 0.0
        self.physics.impact_occurred = False
        self.physics.impact_start_x = None
        self.physics.max_penetration = 0.0
        for b in self.physics.bodies:
            b.angular_vel = np.zeros(3, dtype=np.float32)
            if b.static is not None:
                b.pos[b.static] = b.rest_pos[b.static].copy()
                b.vel[b.static] = 0.0
        self.simulation_started = False
        for b in self.physics.bodies:
            self.rebuild_visuals(b)
        if self.selected:
            self.gizmo.update_position()
        if self.quad_mode:
            self._sync_ortho_views()
        self.time_lbl.setText("Time: 0.0 μs")
        self.sim_speed_lbl.setText(f"Sim Speed: {self.physics.time_scale:.3f}x")
        self.pen_lbl.setText("Pen: 0 mm")

    def on_speed(self, v):
        self.physics.time_scale = v / 1000
        self.speed_lbl.setText(f"{v / 1000:.3f}x")
        self.sim_speed_lbl.setText(f"Sim Speed: {self.physics.time_scale:.3f}x")

    def update_frame(self):
        if not self.running:
            return
        stats = self.physics.step(0.001)
        for b in self.physics.bodies:
            self.update_visuals(b)
        if self.selected:
            self.gizmo.update_position()
        if self.quad_mode:
            self._sync_ortho_views()
        self.time_lbl.setText(f"Time: {stats['time_us']:.1f} μs")
        self.sim_speed_lbl.setText(f"Sim Speed: {self.physics.time_scale:.3f}x")
        self.pen_lbl.setText(f"Pen: {stats['pen']:.1f} mm")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
