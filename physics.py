# -*- coding: utf-8 -*-
"""Physics engine: time stepping, springs, bond breaking, collisions."""

import numpy as np

from materials import MATERIALS


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

        force_mag = a_stiff * strain
        forces = force_mag[:, np.newaxis] * dirs

        ratio = lengths / a_rest
        rep_mask = ratio < compression_limit
        if np.any(rep_mask):
            extra = (compression_limit - ratio[rep_mask]) * a_stiff[rep_mask] * 3.0
            forces[rep_mask] += extra[:, np.newaxis] * dirs[rep_mask]

        force_accum = np.zeros_like(body.pos)
        np.add.at(force_accum, i_idx, forces)
        np.add.at(force_accum, j_idx, -forces)

        if body.active is not None:
            if body.static is not None:
                update = body.active & ~body.static
            else:
                update = body.active
        else:
            update = ~body.static if body.static is not None else np.ones(body.n_particles, dtype=bool)

        nz = update & (body.mass > 0)
        body.vel[nz] += force_accum[nz] / body.mass[nz, np.newaxis] * dt

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
