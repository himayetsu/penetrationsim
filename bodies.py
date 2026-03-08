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
