# -*- coding: utf-8 -*-
"""Transform gizmo: 3D axis arrows and drag-to-translate."""

import numpy as np
from PyQt5.QtGui import QVector3D
import pyqtgraph.opengl as gl


class TransformGizmo:
    AXIS_LENGTH = 0.06
    PICK_THRESHOLD = 30

    def __init__(self, view):
        self.view = view
        self.body = None
        self.arrows = {}
        colors = {'x': (1, 0, 0, 1), 'y': (0, 1, 0, 1), 'z': (0, 0, 1, 1)}
        for axis, color in colors.items():
            item = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=color, width=4, mode='lines')
            item.setVisible(False)
            view.addItem(item)
            self.arrows[axis] = item

    def attach(self, body):
        self.body = body
        self.update_position()
        for item in self.arrows.values():
            item.setVisible(True)

    def detach(self):
        self.body = None
        for item in self.arrows.values():
            item.setVisible(False)

    def update_position(self):
        if self.body is None or self.body.pos is None:
            return
        c = np.mean(self.body.pos, axis=0)
        L = self.AXIS_LENGTH
        self.arrows['x'].setData(pos=np.array([c, c + [L, 0, 0]], dtype=np.float32))
        self.arrows['y'].setData(pos=np.array([c, c + [0, L, 0]], dtype=np.float32))
        self.arrows['z'].setData(pos=np.array([c, c + [0, 0, L]], dtype=np.float32))

    def pick_axis(self, mx, my, view_widget):
        if self.body is None or self.body.pos is None:
            return None
        c = np.mean(self.body.pos, axis=0)
        L = self.AXIS_LENGTH
        endpoints = {
            'x': c + np.array([L, 0, 0]),
            'y': c + np.array([0, L, 0]),
            'z': c + np.array([0, 0, L]),
        }

        view_mat = view_widget.viewMatrix()
        vp = view_widget._default_viewport()
        proj_mat = view_widget.projectionMatrix(vp, vp)
        mvp = proj_mat * view_mat
        w, h = view_widget.width(), view_widget.height()

        def to_screen(pos):
            v = QVector3D(float(pos[0]), float(pos[1]), float(pos[2]))
            clip = mvp.map(v)
            return ((clip.x() * 0.5 + 0.5) * w,
                    (0.5 - clip.y() * 0.5) * h)

        sc = to_screen(c)
        best, best_d = None, self.PICK_THRESHOLD
        for name, ep in endpoints.items():
            se = to_screen(ep)
            d = self._pt_seg_dist(mx, my, sc[0], sc[1], se[0], se[1])
            if d < best_d:
                best_d = d
                best = name
        return best

    @staticmethod
    def _pt_seg_dist(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 < 1e-6:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))
        return np.sqrt((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2)
