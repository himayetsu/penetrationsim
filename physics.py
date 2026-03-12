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
        self.max_penetration = 0.0

    def reset(self):
        self.time = 0.0
        self.impact_occurred = False
        self.impact_start_x = None
        self.max_penetration = 0.0
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

                # Rotational integration: rotate particles around CoM
                omega = body.angular_vel
                omega_sq = float(np.dot(omega, omega))
                if omega_sq > 1e-12 and body.springs is not None and len(body.springs) > 0:
                    connected = np.zeros(body.n_particles, dtype=bool)
                    connected[body.springs[:, 0]] = True
                    connected[body.springs[:, 1]] = True
                    m_conn = body.mass[connected]
                    total_m = float(np.sum(m_conn))
                    if total_m > 1e-12:
                        com = np.sum(body.pos[connected] * m_conn[:, np.newaxis], axis=0) / total_m
                        r = body.pos - com
                        rot_disp = np.cross(omega, r) * sub_dt
                        body.pos[mask] += rot_disp[mask]

        self.time += real_dt
        stats = {'time_us': self.time * 1e6, 'vel': 0, 'pen': 0}
        for b in self.bodies:
            if b.body_type == 'penetrator':
                stats['vel'] = np.mean(np.sqrt(np.sum(b.vel ** 2, axis=1)))
                if self.impact_occurred and self.impact_start_x is not None:
                    current_pen = (np.max(b.pos[:, 0]) - self.impact_start_x) * 1000
                    self.max_penetration = max(self.max_penetration, current_pen)
                    stats['pen'] = self.max_penetration
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

        broken = strain > breaking_strain
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
            forces[rep_mask] -= extra[:, np.newaxis] * dirs[rep_mask]

        # Plastic deformation: permanently update rest length only when a spring is
        # compressed further than its previous plastic state. This prevents rest lengths
        # from collapsing each substep — plasticity ratchets forward, never backward.
        yield_s = mat['yield_strain']
        comp_plastic = strain < -yield_s
        if np.any(comp_plastic):
            p_idx = np.where(active_mask)[0][comp_plastic]
            new_rest = lengths[comp_plastic] / (1.0 - yield_s)
            # Only apply if the spring is being pushed into a MORE compressed state
            # than it has ever been (new rest < current rest = further plastic deformation).
            further = new_rest < body.spring_rest[p_idx]
            if np.any(further):
                body.spring_rest[p_idx[further]] = new_rest[further]

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
            body.angular_vel += (torque / I).astype(np.float32) * dt
            body.angular_vel *= 0.98

    def _handle_collisions(self, dt):
        from scipy.spatial import cKDTree

        bodies = self.bodies
        col_dist = self.collision_dist

        pos_list, vel_list, mass_list = [], [], []
        body_idx_list, part_idx_list = [], []
        body_types = []
        spacings = []
        # Per-body material scalars, expanded to per-particle arrays below
        b_strength, b_restitution, b_youngs, b_friction = [], [], [], []

        for bi, body in enumerate(bodies):
            if body.pos is None:
                continue
            n = body.n_particles
            pos_list.append(body.pos)
            vel_list.append(body.vel)
            mass_list.append(body.mass)
            body_idx_list.append(np.full(n, bi, dtype=np.int32))
            part_idx_list.append(np.arange(n, dtype=np.int32))
            mat = MATERIALS[body.material]
            b_strength.append(np.full(n, mat['strength'], dtype=np.float64))
            b_restitution.append(np.full(n, mat['restitution'], dtype=np.float64))
            b_youngs.append(np.full(n, mat['youngs_modulus'], dtype=np.float64))
            b_friction.append(np.full(n, mat['friction_coeff'], dtype=np.float64))
            body_types.append(body.body_type)
            spacings.append(getattr(body, 'particle_spacing', col_dist))

        if not pos_list:
            return

        all_pos = np.vstack(pos_list).astype(np.float64)
        all_vel = np.vstack(vel_list).astype(np.float64)
        all_mass = np.concatenate(mass_list).astype(np.float64)
        all_body_idx = np.concatenate(body_idx_list)
        all_part_idx = np.concatenate(part_idx_list)
        p_strength = np.concatenate(b_strength)
        p_restitution = np.concatenate(b_restitution)
        p_youngs = np.concatenate(b_youngs)
        p_friction = np.concatenate(b_friction)

        # 1 = penetrator, 0 = armor/other  (for vectorized pen-armor detection)
        p_is_pen = np.array([1 if body_types[b] == 'penetrator' else 0
                              for b in all_body_idx], dtype=np.int8)
        p_spacing = np.array([spacings[b] for b in all_body_idx])

        # --- Pair finding: cKDTree is C-level, far faster than Python cell loop ---
        tree = cKDTree(all_pos)
        pairs = tree.query_pairs(col_dist, output_type='ndarray')

        if len(pairs) == 0:
            return

        ii = pairs[:, 0]
        jj = pairs[:, 1]
        bi_arr = all_body_idx[ii]
        bj_arr = all_body_idx[jj]
        same_body = bi_arr == bj_arr

        # Per-pair effective collision distance (intra-body uses particle spacing)
        eff_cd = np.where(same_body, p_spacing[ii] * 0.85, col_dist)

        diff = all_pos[jj] - all_pos[ii]          # (M, 3)
        dist_sq = np.einsum('ij,ij->i', diff, diff)

        valid = (dist_sq < eff_cd ** 2) & (dist_sq > 1e-12)
        if not np.any(valid):
            return

        ii = ii[valid]; jj = jj[valid]
        bi_arr = bi_arr[valid]; bj_arr = bj_arr[valid]
        same_body = same_body[valid]
        eff_cd = eff_cd[valid]
        diff = diff[valid]
        dist_sq = dist_sq[valid]

        dist = np.sqrt(dist_sq)
        normal = diff / dist[:, np.newaxis]
        overlap = eff_cd - dist

        rel_vel = all_vel[ii] - all_vel[jj]
        vel_normal = np.einsum('ij,ij->i', rel_vel, normal)

        m1 = all_mass[ii]; m2 = all_mass[jj]
        reduced_mass = m1 * m2 / (m1 + m2)
        s1 = p_strength[ii]; s2 = p_strength[jj]
        e1 = p_restitution[ii]; e2 = p_restitution[jj]
        mu = (p_friction[ii] + p_friction[jj]) * 0.5
        E_avg = (p_youngs[ii] + p_youngs[jj]) * 0.5

        ratio1 = s2 / (s1 + s2)
        ratio2 = s1 / (s1 + s2)

        # Impact detection (pen + armor pair with different body indices)
        is_pen_armor = ((p_is_pen[ii] + p_is_pen[jj]) == 1) & (~same_body)
        if np.any(is_pen_armor) and not self.impact_occurred:
            self.impact_occurred = True
            for body in self.bodies:
                if body.body_type == 'penetrator':
                    self.impact_start_x = np.max(body.pos[:, 0])
                    break

        # --- Position corrections (vectorized) ---
        corr = np.where(is_pen_armor, 0.8, 0.5)
        contact_stiff = np.clip(np.sqrt((s1 + s2) / (2.0 * 800e6)), 0.5, 1.5)
        scale_i = (overlap * ratio1 * corr * contact_stiff)[:, np.newaxis]
        scale_j = (overlap * ratio2 * corr * contact_stiff)[:, np.newaxis]
        np.add.at(all_pos, ii, -normal * scale_i)
        np.add.at(all_pos, jj,  normal * scale_j)

        # --- Velocity updates: approaching pairs ---
        app = vel_normal > 0
        if np.any(app):
            a = np.where(app)[0]
            base_e = np.sqrt(e1[a] * e2[a])
            v_ratio = np.minimum(vel_normal[a] / 1000.0, 2.0)
            rest = np.maximum(0.2, base_e * (1.0 - 0.3 * v_ratio ** 2))
            sr = np.minimum(s1[a], s2[a]) / np.maximum(s1[a], s2[a])
            loss = np.minimum(0.5, (1.0 - sr) * 0.4
                              + np.minimum(0.15, mu[a] * vel_normal[a] / 2000.0))
            eff_e = np.maximum(0.2, np.minimum(rest * (1.0 - loss), base_e))
            imp = np.minimum((1.0 + eff_e) * reduced_mass[a] * vel_normal[a],
                             reduced_mass[a] * vel_normal[a] * 1.5)
            imp = np.maximum(0.0, imp)
            intra_scale = np.where(same_body[a],
                                   np.where(p_is_pen[ii[a]] == 1, 0.85, 0.9),
                                   1.0)
            imp *= intra_scale
            impulse = imp[:, np.newaxis] * normal[a]
            np.add.at(all_vel, ii[a], -impulse / m1[a, np.newaxis])
            np.add.at(all_vel, jj[a],  impulse / m2[a, np.newaxis])

        # --- Velocity updates: receding / compressed pairs ---
        rec = ~app
        if np.any(rec):
            r = np.where(rec)[0]
            ca = np.pi * (col_dist * 0.5) ** 2
            s_ov = overlap[r] / np.maximum(eff_cd[r], 1e-6)
            same_r = same_body[r]
            rf_same = np.minimum(E_avg[r] * s_ov * ca * 1e-6, reduced_mass[r] * 200.0)
            rf_diff = np.minimum(overlap[r] * 50.0, reduced_mass[r] * 50.0)
            rf = np.where(same_r, rf_same, rf_diff)
            imp_s = rf[:, np.newaxis] * normal[r] * dt
            mag = np.linalg.norm(imp_s, axis=1, keepdims=True)
            max_vc = (np.where(same_r, 10.0, 5.0) * reduced_mass[r])[:, np.newaxis]
            scale = np.where(mag > 1e-6, np.minimum(1.0, max_vc / np.maximum(mag, 1e-12)), 0.0)
            imp_s *= scale
            np.add.at(all_vel, ii[r], -imp_s / m1[r, np.newaxis])
            np.add.at(all_vel, jj[r],  imp_s / m2[r, np.newaxis])

        # Mark colliding particles active (vectorized per-body grouping)
        for body_id in np.unique(bi_arr):
            body = bodies[body_id]
            if body.active is not None:
                body.active[all_part_idx[ii[bi_arr == body_id]]] = True
                body.active[all_part_idx[jj[bj_arr == body_id]]] = True

        # Write back positions and velocities
        offset = 0
        for body in bodies:
            if body.pos is None:
                continue
            n = body.n_particles
            body.pos = all_pos[offset:offset + n].astype(np.float32)
            body.vel = all_vel[offset:offset + n].astype(np.float32)
            offset += n
