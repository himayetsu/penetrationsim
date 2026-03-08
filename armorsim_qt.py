# -*- coding: utf-8 -*-
# python armorsim_qt.py

"""ArmorSim 3D - Soft Body Penetration Simulation"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QScrollArea, QDockWidget, QCheckBox,
    QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QMatrix4x4, QVector3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# Real-world material data: density (kg/m³), strength (Pa; affects collision overlap
# resolution and impact energy loss), Young's modulus (Pa; sets spring stiffness),
# restitution, hardness, Poisson ratio, friction, breaking_strain, compression_limit.
# Refs: tungsten (AZoM, material-properties.org); RHA (MIL-DTL-12560K, DEF STAN 95-24);
# ceramic armor (alumina/B4C, AZoM, precision-ceramics).
MATERIALS = {
    'tungsten': {
        'name': 'Tungsten',
        'density': 19300,
        'strength': 980e6,
        'youngs_modulus': 411e9,
        'restitution': 0.28,
        'hardness': 7.5,
        'poisson_ratio': 0.28,
        'friction_coeff': 0.4,
        'breaking_strain': 0.05,
        'compression_limit': 0.40,
        'color': np.array([1.0, 0.7, 0.25])
    },
    'steel': {
        'name': 'RHA Steel',
        'density': 7850,
        'strength': 1000e6,
        'youngs_modulus': 210e9,
        'restitution': 0.55,
        'hardness': 5.0,
        'poisson_ratio': 0.29,
        'friction_coeff': 0.6,
        'breaking_strain': 0.15,
        'compression_limit': 0.50,
        'color': np.array([0.5, 0.55, 0.65])
    },
    'ceramic': {
        'name': 'Ceramic',
        'density': 3950,
        'strength': 900e6,
        'youngs_modulus': 370e9,
        'restitution': 0.22,
        'hardness': 9.0,
        'poisson_ratio': 0.22,
        'friction_coeff': 0.5,
        'breaking_strain': 0.008,
        'compression_limit': 0.65,
        'color': np.array([0.95, 0.9, 0.82])
    },
}


class SoftBody:
    def __init__(self, name, body_type, material='steel'):
        self.name = name
        self.body_type = body_type
        self.material = material
        self.selected = False

        self.pos = None
        self.vel = None
        self.rest_pos = None
        self.mass = None
        self.radius = None
        self.active = None
        self.static = None
        self.angular_vel = np.zeros(3, dtype=np.float32)

        self.springs = None
        self.spring_rest = None
        self.spring_stiff = None

        self.scatter = None
        self.lines = None

        self.initial_velocity = 1700.0
        self.particle_size = 0.006

    @property
    def n_particles(self):
        return len(self.pos) if self.pos is not None else 0


def create_cylinder_body(length, diameter, material, spacing=None):
    if spacing is None:
        spacing = 0.012
    mat = MATERIALS[material]
    radius = diameter / 2
    particles = []

    n_axial = int(length * 0.85 / spacing) + 1
    n_radial = max(1, int(radius / spacing))

    for i in range(n_axial):
        x = i * spacing
        particles.append([x, 0, 0])
        for r_idx in range(1, n_radial + 1):
            r = r_idx * spacing
            if r > radius:
                break
            n_circ = max(6, int(2 * np.pi * r / spacing))
            for j in range(n_circ):
                angle = 2 * np.pi * j / n_circ
                particles.append([x, r * np.cos(angle), r * np.sin(angle)])

    nose_length = length * 0.15
    n_nose = int(nose_length / spacing) + 1
    for i in range(1, n_nose + 1):
        t = i / n_nose
        x = length * 0.85 + t * nose_length
        current_r = radius * (1 - t ** 1.2)
        if current_r < spacing * 0.3:
            particles.append([x, 0, 0])
        else:
            particles.append([x, 0, 0])
            n_circ = max(4, int(2 * np.pi * current_r / spacing))
            for j in range(n_circ):
                angle = 2 * np.pi * j / n_circ
                particles.append([x, current_r * np.cos(angle), current_r * np.sin(angle)])

    pos = np.array(particles, dtype=np.float32)
    pos[:, 0] -= 0.1
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float32)
    mass = np.full(n, mat['density'] * spacing**3, dtype=np.float32)

    springs = []
    spring_dist = spacing * 1.8
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos[j] - pos[i])
            if dist < spring_dist:
                springs.append([i, j, dist])

    springs = np.array(springs, dtype=np.float32)
    spring_indices = springs[:, :2].astype(np.int32)
    spring_rest = springs[:, 2]
    area = spacing ** 2
    k = mat['youngs_modulus'] * area / spacing * 1e-7
    spring_stiff = np.full(len(springs), k, dtype=np.float32)
    return pos, vel, mass, spring_indices, spring_rest, spring_stiff


def create_plate_body(width, height, thickness, angle_deg, material, spacing=None):
    if spacing is None:
        spacing = 0.014
    mat = MATERIALS[material]
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    particles = []
    static_particles = []
    nx = max(1, int(round(width / spacing)) + 1)
    ny = max(1, int(round(height / spacing)) + 1)
    nz = max(1, int(round(thickness / spacing)) + 1)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = -width / 2 + ix * spacing
                y = -height / 2 + iy * spacing
                z = iz * spacing
                rx = z * cos_a + x * sin_a
                rz = -z * sin_a + x * cos_a
                particles.append([rx + 0.22, y, rz])
                is_static = (ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1)
                static_particles.append(is_static)

    pos = np.array(particles, dtype=np.float32)
    n = len(pos)
    vel = np.zeros((n, 3), dtype=np.float32)
    mass = np.full(n, mat['density'] * spacing**3, dtype=np.float32)
    static = np.array(static_particles, dtype=bool)

    spring_dist = spacing * 1.8
    grid = {}
    for idx in range(n):
        giz = idx // (nx * ny)
        giy = (idx // nx) % ny
        gix = idx % nx
        grid.setdefault((gix, giy, giz), []).append(idx)

    springs = []
    for i in range(n):
        giz = i // (nx * ny)
        giy = (i // nx) % ny
        gix = i % nx
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j in grid.get((gix + dx, giy + dy, giz + dz), []):
                        if j <= i:
                            continue
                        d = pos[j] - pos[i]
                        dist = np.sqrt(np.dot(d, d))
                        if dist < spring_dist:
                            springs.append([i, j, dist])

    springs = np.array(springs, dtype=np.float32) if springs else np.zeros((0, 3), dtype=np.float32)
    if len(springs) > 0:
        spring_indices = springs[:, :2].astype(np.int32)
        spring_rest = springs[:, 2]
    else:
        spring_indices = np.zeros((0, 2), dtype=np.int32)
        spring_rest = np.zeros(0, dtype=np.float32)

    area = spacing ** 2
    k = mat['youngs_modulus'] * area / spacing * 1e-7 * 0.6
    spring_stiff = np.full(len(springs), k, dtype=np.float32)
    return pos, vel, mass, spring_indices, spring_rest, spring_stiff, static


class Penetrator(SoftBody):
    def __init__(self, name="Penetrator"):
        super().__init__(name, "penetrator", "tungsten")
        self.length = 0.055
        self.diameter = 0.030
        self.initial_velocity = 900.0
        self.particle_size = 0.004
        self.rebuild()

    def rebuild(self):
        pos, vel, mass, springs, rest, stiff = create_cylinder_body(
            self.length, self.diameter, self.material, spacing=0.012)
        self.pos = pos
        self.vel = vel
        self.rest_pos = pos.copy()
        self.mass = mass
        self.springs = springs
        self.spring_rest = rest
        self.spring_stiff = stiff
        self.radius = np.full(len(pos), self.particle_size, dtype=np.float32)
        self.active = np.zeros(len(pos), dtype=bool)
        self.static = None
        self.angular_vel = np.zeros(3, dtype=np.float32)


class ArmorPlate(SoftBody):
    PARTICLE_SPACING = 0.014

    def __init__(self, name="Armor"):
        super().__init__(name, "armor", "steel")
        self.width = 0.18
        self.height = 0.14
        self.thickness = 0.038
        self.angle = 60
        self.particle_size = 0.005
        self.rebuild()

    def rebuild(self):
        pos, vel, mass, springs, rest, stiff, static = create_plate_body(
            self.width, self.height, self.thickness, self.angle,
            self.material, spacing=self.PARTICLE_SPACING)
        self.pos = pos
        self.vel = vel
        self.rest_pos = pos.copy()
        self.mass = mass
        self.static = static
        self.springs = springs
        self.spring_rest = rest
        self.spring_stiff = stiff
        self.radius = np.full(len(pos), self.particle_size, dtype=np.float32)
        self.active = np.zeros(len(pos), dtype=bool)
        self.angular_vel = np.zeros(3, dtype=np.float32)


class PhysicsEngine:
    def __init__(self):
        self.bodies = []
        self.time = 0.0
        self.time_scale = 0.002
        self.collision_dist = 0.012
        self.impact_occurred = False
        self.impact_start_x = None

    def reset(self):
        self.time = 0.0
        self.impact_occurred = False
        self.impact_start_x = None
        for body in self.bodies:
            body.pos = body.rest_pos.copy()
            body.vel.fill(0)
            body.angular_vel = np.zeros(3, dtype=np.float32)

            if body.static is not None:
                body.pos[body.static] = body.rest_pos[body.static].copy()
                body.vel[body.static] = 0.0

            if body.springs is not None and len(body.springs) > 0:
                i_idx = body.springs[:, 0]
                j_idx = body.springs[:, 1]
                delta = body.pos[j_idx] - body.pos[i_idx]
                body.spring_rest = np.linalg.norm(delta, axis=1).copy()

            if body.body_type == 'penetrator':
                body.vel[:, 0] = body.initial_velocity
                if body.active is not None:
                    body.active.fill(True)
            else:
                if body.active is not None:
                    body.active.fill(False)

    def _propagate_active(self, body):
        if body.springs is None or len(body.springs) == 0:
            return
        i_idx = body.springs[:, 0]
        j_idx = body.springs[:, 1]
        for _ in range(5):
            new_i = body.active[j_idx] & ~body.active[i_idx]
            new_j = body.active[i_idx] & ~body.active[j_idx]
            if not (np.any(new_i) or np.any(new_j)):
                break
            if np.any(new_i):
                body.active[i_idx[new_i]] = True
            if np.any(new_j):
                body.active[j_idx[new_j]] = True

    def step(self, dt):
        real_dt = dt * self.time_scale
        substeps = 4
        sub_dt = real_dt / substeps

        for _ in range(substeps):
            for body in self.bodies:
                if body.active is None:
                    continue
                if body.body_type == 'penetrator':
                    body.active.fill(True)
                else:
                    vel_sq = np.sum(body.vel ** 2, axis=1)
                    body.active = vel_sq > 0.01
                if body.static is not None:
                    body.active[body.static] = False
                    body.vel[body.static] = 0.0

            self._handle_collisions(sub_dt)

            for body in self.bodies:
                if body.active is not None:
                    self._propagate_active(body)

            for body in self.bodies:
                self._apply_springs(body, sub_dt)

            for body in self.bodies:
                mask = body.active if body.active is not None else np.ones(body.n_particles, dtype=bool)
                if body.static is not None:
                    mask = mask & ~body.static
                body.pos[mask] += body.vel[mask] * sub_dt

        self.time += real_dt
        stats = {'time_us': self.time * 1e6, 'vel': 0, 'pen': 0}
        for b in self.bodies:
            if b.body_type == 'penetrator':
                stats['vel'] = np.mean(np.sqrt(np.sum(b.vel ** 2, axis=1)))
                if self.impact_occurred and self.impact_start_x is not None:
                    stats['pen'] = (np.max(b.pos[:, 0]) - self.impact_start_x) * 1000
        return stats

    def _apply_springs(self, body, dt):
        if body.springs is None or len(body.springs) == 0:
            return

        if body.active is not None:
            si = body.springs[:, 0]
            sj = body.springs[:, 1]
            active_mask = body.active[si] | body.active[sj]
            if not np.any(active_mask):
                return
        else:
            active_mask = np.ones(len(body.springs), dtype=bool)

        a_springs = body.springs[active_mask]
        a_rest = body.spring_rest[active_mask]
        a_stiff = body.spring_stiff[active_mask]
        i_idx = a_springs[:, 0]
        j_idx = a_springs[:, 1]

        delta = body.pos[j_idx] - body.pos[i_idx]
        lengths = np.linalg.norm(delta, axis=1)
        lengths = np.maximum(lengths, 1e-6)
        dirs = delta / lengths[:, np.newaxis]
        strain = (lengths - a_rest) / a_rest
        abs_strain = np.abs(strain)

        mat = MATERIALS[body.material]
        breaking_strain = mat['breaking_strain']
        compression_limit = mat['compression_limit']

        broken = abs_strain > breaking_strain
        if np.any(broken):
            broken_full = np.where(active_mask)[0][np.where(broken)[0]]
            keep = np.ones(len(body.springs), dtype=bool)
            keep[broken_full] = False
            body.springs = body.springs[keep]
            body.spring_rest = body.spring_rest[keep]
            body.spring_stiff = body.spring_stiff[keep]

            if len(body.springs) == 0:
                body.angular_vel = np.zeros(3, dtype=np.float32)
                return

            if body.active is not None:
                si2 = body.springs[:, 0]
                sj2 = body.springs[:, 1]
                active_mask = body.active[si2] | body.active[sj2]
            else:
                active_mask = np.ones(len(body.springs), dtype=bool)
            if not np.any(active_mask):
                return

            a_springs = body.springs[active_mask]
            a_rest = body.spring_rest[active_mask]
            a_stiff = body.spring_stiff[active_mask]
            i_idx = a_springs[:, 0]
            j_idx = a_springs[:, 1]
            delta = body.pos[j_idx] - body.pos[i_idx]
            lengths = np.linalg.norm(delta, axis=1)
            lengths = np.maximum(lengths, 1e-6)
            dirs = delta / lengths[:, np.newaxis]
            strain = (lengths - a_rest) / a_rest
            abs_strain = np.abs(strain)

        # Spring forces: standard soft-body F = k * strain * direction
        force_mag = a_stiff * strain
        forces = force_mag[:, np.newaxis] * dirs

        ratio = lengths / a_rest
        rep_mask = ratio < compression_limit
        if np.any(rep_mask):
            extra = (compression_limit - ratio[rep_mask]) * a_stiff[rep_mask] * 3.0
            forces[rep_mask] += extra[:, np.newaxis] * dirs[rep_mask]

        # Accumulate forces per particle
        force_accum = np.zeros_like(body.pos)
        np.add.at(force_accum, i_idx, forces)
        np.add.at(force_accum, j_idx, -forces)

        # Apply forces to velocities (standard soft body)
        if body.active is not None:
            if body.static is not None:
                update = body.active & ~body.static
            else:
                update = body.active
        else:
            update = ~body.static if body.static is not None else np.ones(body.n_particles, dtype=bool)

        nz = update & (body.mass > 0)
        body.vel[nz] += force_accum[nz] / body.mass[nz, np.newaxis] * dt

        # Damping only for particles still connected to the locked (static) boundary
        if body.body_type == 'armor' and body.static is not None and len(body.springs) > 0:
            anchored = body.static.copy()
            si = body.springs[:, 0]
            sj = body.springs[:, 1]
            for _ in range(body.n_particles):
                prev = anchored.copy()
                anchored[si] |= anchored[sj]
                anchored[sj] |= anchored[si]
                if np.all(anchored == prev):
                    break
            damp_mask = nz & anchored
            body.vel[damp_mask] *= 0.98

        # Track angular velocity from spring torques (secondary)
        connected = np.zeros(body.n_particles, dtype=bool)
        if len(body.springs) > 0:
            connected[body.springs[:, 0]] = True
            connected[body.springs[:, 1]] = True
        m_conn = body.mass[connected]
        total_m = np.sum(m_conn)
        if total_m > 1e-12:
            com = np.sum(body.pos[connected] * m_conn[:, np.newaxis], axis=0) / total_m
            r = body.pos - com
            torque = np.sum(np.cross(r, force_accum), axis=0)
            I = max(np.sum(m_conn * np.sum((body.pos[connected] - com) ** 2, axis=1)), 1e-12)
            body.angular_vel = (torque / I).astype(np.float32) * dt * 0.99

    def _get_armor_surface_normal(self, armor_body):
        if armor_body.body_type != 'armor':
            return None
        if hasattr(armor_body, 'angle'):
            angle_rad = np.radians(armor_body.angle)
            return np.array([-np.cos(angle_rad), 0.0, np.sin(angle_rad)], dtype=np.float32)
        return np.array([-1, 0, 0], dtype=np.float32)

    def _handle_collisions(self, dt):
        bodies = self.bodies
        col_dist = self.collision_dist

        all_particles = []
        all_velocities = []
        all_masses = []
        all_body_indices = []
        all_particle_indices = []
        all_materials = []
        all_body_types = []

        for bi, body in enumerate(bodies):
            if body.pos is None:
                continue
            n = body.n_particles
            all_particles.append(body.pos)
            all_velocities.append(body.vel)
            all_masses.append(body.mass)
            all_body_indices.append(np.full(n, bi, dtype=np.int32))
            all_particle_indices.append(np.arange(n, dtype=np.int32))
            mat = MATERIALS[body.material]
            all_materials.append({
                'hardness': mat['hardness'], 'strength': mat['strength'],
                'restitution': mat['restitution'], 'youngs_modulus': mat['youngs_modulus'],
                'friction_coeff': mat['friction_coeff'], 'density': mat['density']
            })
            all_body_types.append(body.body_type)

        if not all_particles:
            return

        colliding_particles = set()
        all_pos = np.vstack(all_particles)
        all_vel = np.vstack(all_velocities)
        all_mass = np.concatenate(all_masses)
        all_body_idx = np.concatenate(all_body_indices)
        all_part_idx = np.concatenate(all_particle_indices)
        n_total = len(all_pos)

        cell_size = col_dist * 1.5
        col_dist_sq = col_dist ** 2
        cells = {}
        cell_coords = np.floor(all_pos / cell_size).astype(np.int32)
        for i in range(n_total):
            key = tuple(cell_coords[i])
            cells.setdefault(key, []).append(i)

        for i in range(n_total):
            body_i = all_body_idx[i]
            cell = tuple(cell_coords[i])
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nc = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        if nc not in cells:
                            continue
                        for j in cells[nc]:
                            if i >= j:
                                continue
                            body_j = all_body_idx[j]
                            diff = all_pos[j] - all_pos[i]
                            dist_sq = np.dot(diff, diff)

                            rel_vel = all_vel[i] - all_vel[j]
                            vel_sq = np.dot(rel_vel, rel_vel)

                            will_collide = False
                            if dist_sq < col_dist_sq and dist_sq > 1e-12:
                                will_collide = True
                            elif vel_sq > 100.0:
                                fd = diff + rel_vel * dt
                                if np.dot(fd, fd) < col_dist_sq:
                                    will_collide = True
                                    diff = fd
                                    dist_sq = np.dot(fd, fd)

                            if not will_collide:
                                continue

                            dist = np.sqrt(dist_sq)
                            colliding_particles.add((body_i, all_part_idx[i]))
                            colliding_particles.add((body_j, all_part_idx[j]))

                            normal = diff / dist
                            overlap = col_dist - dist
                            vel_normal = np.dot(rel_vel, normal)

                            mat_i = all_materials[body_i]
                            mat_j = all_materials[body_j]
                            s1, s2 = mat_i['strength'], mat_j['strength']
                            e1, e2 = mat_i['restitution'], mat_j['restitution']
                            m1, m2 = all_mass[i], all_mass[j]
                            reduced_mass = (m1 * m2) / (m1 + m2)

                            ratio1 = s2 / (s1 + s2)
                            ratio2 = s1 / (s1 + s2)

                            is_pen_armor = (
                                (all_body_types[body_i] == 'penetrator' and all_body_types[body_j] == 'armor') or
                                (all_body_types[body_i] == 'armor' and all_body_types[body_j] == 'penetrator'))

                            if is_pen_armor and not self.impact_occurred:
                                self.impact_occurred = True
                                for body in self.bodies:
                                    if body.body_type == 'penetrator':
                                        self.impact_start_x = np.max(body.pos[:, 0])
                                        break

                            corr = 0.8 if is_pen_armor else 0.5
                            strength_ref = 800e6
                            contact_stiff = np.sqrt((s1 + s2) / (2.0 * strength_ref))
                            contact_stiff = np.clip(contact_stiff, 0.5, 1.5)
                            all_pos[i] -= normal * overlap * ratio1 * corr * contact_stiff
                            all_pos[j] += normal * overlap * ratio2 * corr * contact_stiff

                            if vel_normal > 0:
                                base_e = np.sqrt(e1 * e2)
                                v_ratio = min(vel_normal / 1000.0, 2.0)
                                rest = max(0.2, base_e * (1 - 0.3 * v_ratio ** 2))
                                mu = (mat_i['friction_coeff'] + mat_j['friction_coeff']) / 2
                                s_ratio = min(s1, s2) / max(s1, s2)
                                loss = min(0.5, (1 - s_ratio) * 0.4 + min(0.15, mu * vel_normal / 2000.0))
                                eff_e = max(0.2, min(rest * (1 - loss), base_e))
                                imp = min((1 + eff_e) * reduced_mass * vel_normal, reduced_mass * vel_normal * 1.5)
                                imp = max(0.0, imp)
                                if body_i == body_j:
                                    imp *= 0.9 if all_body_types[body_i] == 'armor' else 0.85
                                impulse = imp * normal
                                all_vel[i] -= impulse / m1
                                all_vel[j] += impulse / m2
                            else:
                                if body_i == body_j:
                                    E_avg = (mat_i['youngs_modulus'] + mat_j['youngs_modulus']) / 2
                                    ca = np.pi * (col_dist / 2) ** 2
                                    s = overlap / max(col_dist, 1e-6)
                                    rf = min(E_avg * s * ca * 1e-6, reduced_mass * 200.0)
                                else:
                                    rf = min(overlap * 50.0, reduced_mass * 50.0)
                                imp_s = rf * normal * dt
                                max_vc = 10.0 if body_i == body_j else 5.0
                                mag = np.linalg.norm(imp_s / reduced_mass)
                                if mag > 1e-6:
                                    imp_s = imp_s / np.linalg.norm(imp_s) * min(mag, max_vc) * reduced_mass
                                all_vel[i] -= imp_s / m1
                                all_vel[j] += imp_s / m2

        for body_idx, particle_idx in colliding_particles:
            if body_idx < len(bodies) and bodies[body_idx].active is not None:
                if particle_idx < len(bodies[body_idx].active):
                    bodies[body_idx].active[particle_idx] = True

        offset = 0
        for body in bodies:
            if body.pos is None:
                continue
            n = body.n_particles
            body.vel = all_vel[offset:offset + n].copy()
            body.pos = all_pos[offset:offset + n].copy()
            offset += n


# ---------------------------------------------------------------------------
#  Transform Gizmo
# ---------------------------------------------------------------------------

class TransformGizmo:
    AXIS_LENGTH = 0.06
    PICK_THRESHOLD = 30  # pixels

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
        """Pick closest axis arrow to screen coords (mx, my)."""
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


# ---------------------------------------------------------------------------
#  Custom GL View (ortho/perspective, gizmo interaction)
# ---------------------------------------------------------------------------

class SimViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None, mode='perspective'):
        super().__init__(parent)
        self.ortho = mode != 'perspective'
        self.view_mode = mode
        self.gizmo = None
        self.setFocusPolicy(Qt.StrongFocus)

        self._drag_axis = None
        self._drag_start = None
        self._drag_body = None
        self._drag_pos0 = None
        self._drag_rest0 = None

        self._view_label = QLabel(mode.capitalize() if mode != 'perspective' else 'Perspective', self)
        self._view_label.setStyleSheet(
            "color: #6af; background: rgba(0,0,0,120); font-size: 11px; padding: 3px 6px;")
        self._view_label.move(6, 6)

        if mode == 'front':
            self.setCameraPosition(distance=0.5, elevation=0, azimuth=90)
        elif mode == 'right':
            self.setCameraPosition(distance=0.5, elevation=0, azimuth=0)
        elif mode == 'top':
            self.setCameraPosition(distance=0.5, elevation=90, azimuth=90)
        else:
            self.setCameraPosition(distance=0.5, elevation=15, azimuth=90)

    def _default_viewport(self):
        dpr = self.devicePixelRatio()
        return (0, 0, int(self.width() * dpr), int(self.height() * dpr))

    def projectionMatrix(self, region=None, viewport=None):
        if viewport is None:
            viewport = self._default_viewport()
        if region is None:
            region = viewport

        if not self.ortho:
            return super().projectionMatrix(region, viewport)

        x0, y0, w, h = viewport
        w = max(w, 1)
        h = max(h, 1)
        dist = self.opts['distance']
        half_h = dist * 0.5
        half_w = half_h * w / h

        left   = half_w * ((region[0] - x0) * (2.0 / w) - 1)
        right  = half_w * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
        bottom = half_h * ((region[1] - y0) * (2.0 / h) - 1)
        top    = half_h * ((region[1] + region[3] - y0) * (2.0 / h) - 1)

        tr = QMatrix4x4()
        tr.ortho(left, right, bottom, top, dist * 0.001, dist * 1000.)
        return tr

    def orbit(self, azim, elev):
        if self.ortho:
            return
        super().orbit(azim, elev)

    def mousePressEvent(self, ev):
        self.setFocus()
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

        if ev.button() == Qt.LeftButton and self.gizmo and self.gizmo.body:
            axis = self.gizmo.pick_axis(ev.x(), ev.y(), self)
            if axis:
                self._drag_axis = axis
                self._drag_start = (ev.x(), ev.y())
                self._drag_body = self.gizmo.body
                self._drag_pos0 = self.gizmo.body.pos.copy()
                self._drag_rest0 = self.gizmo.body.rest_pos.copy()
                ev.accept()
                return

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if self._drag_axis:
            dx = ev.x() - self._drag_start[0]
            dy = ev.y() - self._drag_start[1]
            world_delta = self._compute_drag_delta(dx, dy, self._drag_axis)
            offset = np.zeros(3, dtype=np.float32)
            idx = {'x': 0, 'y': 1, 'z': 2}[self._drag_axis]
            offset[idx] = world_delta
            self._drag_body.pos = self._drag_pos0 + offset
            self._drag_body.rest_pos = self._drag_rest0 + offset
            if self.gizmo:
                self.gizmo.update_position()
            self.update()
            ev.accept()
            return

        if ev.buttons() == Qt.RightButton:
            if ev.modifiers() & Qt.ControlModifier:
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == Qt.MiddleButton:
            if ev.modifiers() & Qt.ControlModifier:
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')

    def mouseReleaseEvent(self, ev):
        if self._drag_axis:
            body = self._drag_body
            self._drag_axis = None
            if body and getattr(self, 'main_window', None):
                self.main_window.update_visuals(body)
                if self.main_window.quad_mode:
                    self.main_window._sync_ortho_views()
            ev.accept()
            return

    def _compute_drag_delta(self, mouse_dx, mouse_dy, axis):
        center = np.mean(self._drag_pos0, axis=0)
        axis_dir = {'x': np.array([1, 0, 0], dtype=np.float64),
                     'y': np.array([0, 1, 0], dtype=np.float64),
                     'z': np.array([0, 0, 1], dtype=np.float64)}[axis]
        ref = 0.01
        view_mat = self.viewMatrix()
        vp = self._default_viewport()
        proj_mat = self.projectionMatrix(vp, vp)
        mvp = proj_mat * view_mat
        w, h = self.width(), self.height()

        def to_screen(pos):
            v = QVector3D(float(pos[0]), float(pos[1]), float(pos[2]))
            c = mvp.map(v)
            return np.array([(c.x() * 0.5 + 0.5) * w, (0.5 - c.y() * 0.5) * h])

        s0 = to_screen(center)
        s1 = to_screen(center + axis_dir * ref)
        screen_dir = s1 - s0
        screen_len = np.linalg.norm(screen_dir)
        if screen_len < 0.1:
            return 0.0
        screen_dir /= screen_len
        ppu = screen_len / ref
        projected_px = np.dot(np.array([mouse_dx, mouse_dy], dtype=np.float64), screen_dir)
        return float(projected_px / ppu)

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_1:
            self.ortho = True
            self.view_mode = 'front'
            self.setCameraPosition(elevation=0, azimuth=90)
        elif key == Qt.Key_3:
            self.ortho = True
            self.view_mode = 'right'
            self.setCameraPosition(elevation=0, azimuth=0)
        elif key == Qt.Key_7:
            self.ortho = True
            self.view_mode = 'top'
            self.setCameraPosition(elevation=90, azimuth=90)
        elif key == Qt.Key_5:
            self.ortho = not self.ortho
        elif key == Qt.Key_0:
            self.ortho = False
            self.view_mode = 'perspective'
            self.setCameraPosition(distance=0.5, elevation=15, azimuth=90)
        else:
            super().keyPressEvent(ev)
            return
        lbl = self.view_mode.capitalize() if self.view_mode != 'perspective' else 'Perspective'
        if self.ortho and self.view_mode == 'perspective':
            lbl = 'Ortho'
        self._view_label.setText(lbl)
        self.update()


# ---------------------------------------------------------------------------
#  2D Orthographic Canvas
# ---------------------------------------------------------------------------

PROJ_AXES = {
    'front': (2, 1, 'Z', 'Y'),
    'right': (0, 1, 'X', 'Y'),
    'top':   (0, 2, 'X', 'Z'),
}


class OrthoCanvas(QWidget):
    """Projects 3D particle positions onto a 2D scatter canvas."""

    def __init__(self, mode='front', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.bodies = []
        self.show_springs = True
        self.setMinimumSize(100, 100)
        self.setStyleSheet("background: #16181c;")

        self._label = QLabel(mode.capitalize(), self)
        self._label.setStyleSheet(
            "color: #6af; background: rgba(0,0,0,120); font-size: 11px; padding: 3px 6px;")
        self._label.move(6, 6)

        self._pan = np.array([0.0, 0.0])
        self._zoom = 1.0
        self._last_mouse = None

    def set_bodies(self, bodies):
        self.bodies = bodies
        self.update()

    def paintEvent(self, ev):
        from PyQt5.QtGui import QPainter, QPen, QBrush
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(22, 24, 28))

        w, h = self.width(), self.height()
        cx, cy = w / 2 + self._pan[0], h / 2 + self._pan[1]
        ax_h, ax_v, lbl_h, lbl_v = PROJ_AXES[self.mode]
        scale = min(w, h) * 2.5 * self._zoom

        grid_pen = QPen(QColor(35, 35, 35, 120))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        grid_step = 0.04 * scale
        if grid_step > 4:
            for i in range(-30, 31):
                gx = cx + i * grid_step
                gy = cy + i * grid_step
                if 0 <= gx <= w:
                    p.drawLine(int(gx), 0, int(gx), h)
                if 0 <= gy <= h:
                    p.drawLine(0, int(gy), w, int(gy))

        for body in self.bodies:
            if body.pos is None:
                continue
            mat = MATERIALS[body.material]
            base_color = mat['color']

            pos_h = body.pos[:, ax_h]
            pos_v = body.pos[:, ax_v]
            sx = cx + pos_h * scale
            sy = cy - pos_v * scale

            if self.show_springs and body.springs is not None and len(body.springs) > 0:
                lc = QColor(int(base_color[0] * 153), int(base_color[1] * 153),
                            int(base_color[2] * 153), 60)
                spring_pen = QPen(lc)
                spring_pen.setWidth(1)
                p.setPen(spring_pen)
                springs = body.springs
                for k in range(len(springs)):
                    i, j = springs[k]
                    p.drawLine(int(sx[i]), int(sy[i]), int(sx[j]), int(sy[j]))

            vel_mag = np.sqrt(np.sum(body.vel ** 2, axis=1))
            vel_norm = np.clip(vel_mag / 1800, 0, 1)

            radius = 3
            for k in range(len(sx)):
                vn = vel_norm[k]
                if vn < 0.2:
                    t = vn * 5
                    r = base_color[0] * (1 - t) + 0.2 * t
                    g = base_color[1] * (1 - t) + 0.5 * t
                    b = base_color[2] * (1 - t) + 1.0 * t
                elif vn < 0.4:
                    r, g, b = 0.2, 0.8, 1.0
                elif vn < 0.6:
                    r, g, b = 0.2, 1.0, 0.4
                elif vn < 0.8:
                    r, g, b = 1.0, 1.0, 0.2
                else:
                    r, g, b = 1.0, 0.3, 0.1
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(color))
                p.drawEllipse(int(sx[k] - radius), int(sy[k] - radius),
                              radius * 2, radius * 2)

        axis_pen = QPen(QColor(100, 100, 100, 180))
        axis_pen.setWidth(1)
        p.setPen(axis_pen)
        p.drawLine(int(cx), 0, int(cx), h)
        p.drawLine(0, int(cy), w, int(cy))

        label_font = p.font()
        label_font.setPointSize(9)
        p.setFont(label_font)
        p.setPen(QColor(80, 80, 80))
        p.drawText(w - 18, int(cy) - 4, lbl_h)
        p.drawText(int(cx) + 4, 14, lbl_v)

        p.end()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        factor = 1.001 ** delta
        self._zoom *= factor
        self._zoom = max(0.1, min(50.0, self._zoom))
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton or ev.button() == Qt.MiddleButton:
            self._last_mouse = (ev.x(), ev.y())

    def mouseMoveEvent(self, ev):
        if self._last_mouse is not None:
            dx = ev.x() - self._last_mouse[0]
            dy = ev.y() - self._last_mouse[1]
            self._pan[0] += dx
            self._pan[1] += dy
            self._last_mouse = (ev.x(), ev.y())
            self.update()

    def mouseReleaseEvent(self, ev):
        self._last_mouse = None


# ---------------------------------------------------------------------------
#  Main Window
# ---------------------------------------------------------------------------

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

        self.setup_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.create_scene()

    # ---- UI setup ----

    def setup_ui(self):
        # Central view container
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

        # Scene dock
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

        # Properties dock
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

        # Simulation dock
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

        row2.addStretch()

        tl_l.addLayout(row1)
        tl_l.addLayout(row2)
        tl_dock.setWidget(tl_w)
        self.addDockWidget(Qt.BottomDockWidgetArea, tl_dock)

        self._armor_rebuild_timer = QTimer(self)
        self._armor_rebuild_timer.setSingleShot(True)
        self._armor_rebuild_body = None
        self._armor_rebuild_timer.timeout.connect(self._do_armor_rebuild)

    # ---- View controls ----

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

    # ---- Quad view ----

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

    # ---- Scene management ----

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

    def update_visuals(self, body):
        if body.scatter is None:
            return
        mat = MATERIALS[body.material]
        colors = np.zeros((body.n_particles, 4), dtype=np.float32)
        vel_mag = np.sqrt(np.sum(body.vel ** 2, axis=1))
        vel_norm = np.clip(vel_mag / 1800, 0, 1)
        mc = mat['color']

        m1 = vel_norm < 0.2
        m2 = (vel_norm >= 0.2) & (vel_norm < 0.4)
        m3 = (vel_norm >= 0.4) & (vel_norm < 0.6)
        m4 = (vel_norm >= 0.6) & (vel_norm < 0.8)
        m5 = vel_norm >= 0.8
        colors[m1, :3] = mc * (1 - vel_norm[m1, np.newaxis] * 2) + np.array([0.2, 0.5, 1.0]) * (vel_norm[m1, np.newaxis] * 2)
        colors[m2, :3] = [0.2, 0.8, 1.0]
        colors[m3, :3] = [0.2, 1.0, 0.4]
        colors[m4, :3] = [1.0, 1.0, 0.2]
        colors[m5, :3] = [1.0, 0.3, 0.1]
        colors[:, 3] = 1.0
        if body.selected:
            colors[:, :3] = np.clip(colors[:, :3] + 0.1, 0, 1)
        body.scatter.setData(pos=body.pos, color=colors)

        if body.lines is not None:
            if body.springs is not None and len(body.springs) > 0 and self.show_springs:
                lp = np.zeros((len(body.springs) * 2, 3), dtype=np.float32)
                for k, (i, j) in enumerate(body.springs):
                    lp[k * 2] = body.pos[i]
                    lp[k * 2 + 1] = body.pos[j]
                body.lines.setData(pos=lp)
                body.lines.setVisible(True)
            else:
                body.lines.setVisible(False)

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

    # ---- Selection / properties ----

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
        if body.body_type == 'penetrator':
            mat_cb.addItems(['Tungsten', 'RHA Steel'])
        else:
            mat_cb.addItems(['RHA Steel', 'Ceramic'])
        mat_cb.setCurrentText(MATERIALS[body.material]['name'])
        mat_cb.currentTextChanged.connect(lambda t: self.change_mat(body, t))
        ml.addWidget(mat_cb)
        self.props_box.addWidget(mat_grp)

        dim = QGroupBox("Dimensions")
        dl = QVBoxLayout(dim)
        if body.body_type == 'penetrator':
            self._spin("Length (mm)", 1, 10000, int(body.length * 1000),
                       lambda v: self.change_dim(body, 'length', v / 1000), dl)
            self._spin("Diameter (mm)", 1, 1000, int(body.diameter * 1000),
                       lambda v: self.change_dim(body, 'diameter', v / 1000), dl)
            self._spin("Velocity (m/s)", 1, 10000, int(body.initial_velocity),
                       lambda v: setattr(body, 'initial_velocity', float(v)), dl, 50)
        else:
            self._spin("Width (mm)", 1, 10000, int(body.width * 1000),
                       lambda v: self.change_dim(body, 'width', v / 1000), dl)
            self._spin("Height (mm)", 1, 10000, int(body.height * 1000),
                       lambda v: self.change_dim(body, 'height', v / 1000), dl)
            self._spin("Thickness (mm)", 1, 10000, int(body.thickness * 1000),
                       lambda v: self.change_dim(body, 'thickness', v / 1000), dl)
            self._spin("Angle (°)", 0, 90, body.angle,
                       lambda v: self.change_dim(body, 'angle', v), dl)
        self.props_box.addWidget(dim)

    def _spin(self, label, mn, mx, val, cb, layout, step=1):
        h = QHBoxLayout()
        h.addWidget(QLabel(label))
        s = QSpinBox()
        s.setRange(mn, mx)
        s.setValue(val)
        s.setSingleStep(step)
        s.valueChanged.connect(cb)
        h.addWidget(s)
        layout.addLayout(h)

    def clear_props(self):
        while self.props_box.count():
            w = self.props_box.takeAt(0).widget()
            if w:
                w.deleteLater()

    def change_mat(self, body, name):
        m = {'Tungsten': 'tungsten', 'RHA Steel': 'steel', 'Ceramic': 'ceramic'}
        body.material = m.get(name, 'steel')
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

    # ---- Simulation controls ----

    def toggle_play(self):
        self.running = self.play_btn.isChecked()
        if self.running:
            self.play_btn.setText("⏸ Pause")
            if not self.simulation_started:
                for b in self.physics.bodies:
                    b._reset_rest_pos = b.rest_pos.copy()
                    b._reset_springs = b.springs.copy() if b.springs is not None else None
                    b._reset_spring_rest = b.spring_rest.copy() if b.spring_rest is not None else None
                    b._reset_spring_stiff = b.spring_stiff.copy() if b.spring_stiff is not None else None
                self.physics.reset()
                for b in self.physics.bodies:
                    self.update_visuals(b)
                self.simulation_started = True
            self.timer.start(16)
        else:
            self.play_btn.setText("▶ Play")
            self.timer.stop()

    def reset_sim(self):
        self.running = False
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶ Play")
        self.timer.stop()
        for b in self.physics.bodies:
            if hasattr(b, '_reset_rest_pos'):
                b.rest_pos = b._reset_rest_pos.copy()
                b.pos = b._reset_rest_pos.copy()
                if b._reset_springs is not None:
                    b.springs = b._reset_springs.copy()
                    b.spring_rest = b._reset_spring_rest.copy()
                    b.spring_stiff = b._reset_spring_stiff.copy()
        self.physics.reset()
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


if __name__ == '__main__':
    main()
