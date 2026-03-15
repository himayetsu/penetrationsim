# -*- coding: utf-8 -*-
"""Soft body types and geometry: SoftBody, penetrator/plate creation, Penetrator, ArmorPlate."""

import numpy as np

from materials import MATERIALS


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
        self.spring_rest_original = None  # geometric rest lengths, never updated by plasticity
        self.spring_stiff = None

        self.scatter = None
        self.lines = None

        self.initial_velocity = 1700.0
        self.particle_size = 0.006
        self.particle_spacing = 0.012

    @property
    def n_particles(self):
        return len(self.pos) if self.pos is not None else 0


def create_cylinder_body(length, diameter, material, spacing=None):
    if spacing is None:
        spacing = float(np.clip(diameter / 2.5, 0.007, 0.012))
    mat = MATERIALS[material]
    radius = diameter / 2
    particles = []

    # Nose fraction scales with diameter: wider rods need longer tapers
    # 20mm→20%, 50mm→30%, 122mm→50%
    nose_fraction = float(np.clip(0.20 + (diameter - 0.020) * 2.941, 0.15, 0.55))
    body_fraction = 1.0 - nose_fraction

    # Only add nose rings if the body itself has rings (avoids unsupported ring particles
    # on thin rods where radius < spacing, which causes explosion on small calibres)
    body_has_rings = (spacing <= radius)

    n_axial = int(length * body_fraction / spacing) + 1
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

    body_end_x = (n_axial - 1) * spacing
    # Nose starts exactly where body ends — no gap, guaranteed spring connectivity
    nose_length = length - body_end_x
    n_nose = max(1, int(nose_length / spacing) + 1)

    for i in range(1, n_nose + 1):
        t = i / n_nose  # 0 → 1 (body end → tip)
        x = body_end_x + t * nose_length
        current_r = radius * (1 - t ** 1.3)
        particles.append([x, 0, 0])
        if body_has_rings and current_r >= spacing * 0.4:
            # Inner rings at spacing multiples, outermost ring at actual taper radius
            # so the outer surface shrinks smoothly (shorter rest lengths = weaker springs naturally)
            n_inner = max(0, int(current_r / spacing) - 1)
            for r_idx in range(1, n_inner + 1):
                r = r_idx * spacing
                n_circ = max(4, int(2 * np.pi * r / spacing))
                for j in range(n_circ):
                    angle = 2 * np.pi * j / n_circ
                    particles.append([x, r * np.cos(angle), r * np.sin(angle)])
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
    k_per_unit = mat['youngs_modulus'] * mat.get('stiffness_scale', 6e-4)
    # spring_stiff = k * rest_length: shorter nose springs are naturally weaker, no extra scaling needed
    spring_stiff = (k_per_unit * spring_rest).astype(np.float32)
    shear_boost = mat.get('shear_boost', 2.5)
    diag_mask = spring_rest > spacing * 1.1
    spring_stiff[diag_mask] *= shear_boost
    return pos, vel, mass, spring_indices, spring_rest, spring_stiff


def create_plate_body(width, height, thickness, angle_deg, material, spacing=None):
    if spacing is None:
        spacing = float(np.clip(min(thickness / 2.5, width / 10.0, height / 10.0), 0.007, 0.012))
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

    k_per_unit = mat['youngs_modulus'] * mat.get('stiffness_scale', 6e-4)
    spring_stiff = (k_per_unit * spring_rest).astype(np.float32)
    shear_boost = mat.get('shear_boost', 2.2)
    diag_mask = spring_rest > spacing * 1.1
    spring_stiff[diag_mask] *= shear_boost
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
        saved_center = np.mean(self.rest_pos, axis=0) if self.rest_pos is not None else None
        spacing = float(np.clip(self.diameter / 2.5, 0.007, 0.012))
        self.particle_spacing = spacing
        pos, vel, mass, springs, rest, stiff = create_cylinder_body(
            self.length, self.diameter, self.material, spacing=spacing)
        self.pos = pos
        self.vel = vel
        self.rest_pos = pos.copy()
        self.mass = mass
        self.springs = springs
        self.spring_rest = rest
        self.spring_rest_original = rest.copy()
        self.spring_stiff = stiff
        self.radius = np.full(len(pos), self.particle_size, dtype=np.float32)
        self.active = np.zeros(len(pos), dtype=bool)
        self.static = None
        self.angular_vel = np.zeros(3, dtype=np.float32)
        if saved_center is not None:
            offset = saved_center - np.mean(self.rest_pos, axis=0)
            self.pos += offset
            self.rest_pos += offset


class ArmorPlate(SoftBody):
    def __init__(self, name="Armor"):
        super().__init__(name, "armor", "steel")
        self.width = 0.18
        self.height = 0.14
        self.thickness = 0.038
        self.angle = 60
        self.particle_size = 0.005
        self.rebuild()

    def rebuild(self):
        saved_center = np.mean(self.rest_pos, axis=0) if self.rest_pos is not None else None
        spacing = float(np.clip(min(self.thickness / 2.5, self.width / 10.0, self.height / 10.0), 0.007, 0.012))
        self.particle_spacing = spacing
        pos, vel, mass, springs, rest, stiff, static = create_plate_body(
            self.width, self.height, self.thickness, self.angle, self.material, spacing=spacing)
        self.pos = pos
        self.vel = vel
        self.rest_pos = pos.copy()
        self.mass = mass
        self.static = static
        self.springs = springs
        self.spring_rest = rest
        self.spring_rest_original = rest.copy()
        self.spring_stiff = stiff
        self.radius = np.full(len(pos), self.particle_size, dtype=np.float32)
        self.active = np.zeros(len(pos), dtype=bool)
        self.angular_vel = np.zeros(3, dtype=np.float32)
        if saved_center is not None:
            offset = saved_center - np.mean(self.rest_pos, axis=0)
            self.pos += offset
            self.rest_pos += offset
