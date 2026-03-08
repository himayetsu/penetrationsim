# ArmorSim 3D

3D soft-body armor penetration simulation: spring–mass particles, material-based bond breaking, and collision response. Single-file Qt application.

**Run:** `pip install -r requirements.txt` then `python armorsim_qt.py`

---

## Technologies

- **Python 3.8+** – Core logic and numerics
- **NumPy** – Vectorized particle/spring arrays and linear algebra
- **PyQt5** – Main window, docks, controls, 2D ortho canvases (QPainter)
- **PyQtGraph (OpenGL)** – 3D viewport (`GLViewWidget`), scatter plots, line plots, custom ortho projection
- **OpenGL** – Via PyQtGraph / PyOpenGL for 3D rendering

---

## Physics Model

### Soft bodies

Bodies (penetrator, armor) are particle meshes. Each particle has position **x**, velocity **v**, mass **m** (from density × spacing³). Springs connect nearby particles (rest length **L₀**, stiffness **k**).

**Spring force (per spring):**

- **ε** = strain = (L − L₀) / L₀  
- **F** = k · ε · **ê**  (ê = unit vector along the spring)

**Stiffness** (from Young’s modulus *E* and spacing):

- k = E · (spacing² / spacing) · 1e−7  (penetrator)  
- Armor: same form with factor 0.6

Particles within `spacing × 1.8` are connected. Penetrator uses all pairs within that distance; armor uses a 3D grid so only 27-neighbor cells are checked (O(N) in particles).

### Bond breaking

- If |ε| > **breaking_strain** (material property), that spring is removed from the body (no force, not drawn).
- No explicit “fragment” objects: disconnected clusters simply stop sharing springs and are no longer damped if they were armor (see below).

### Compression limit (repulsion)

- **r** = L / L₀  
- If **r < compression_limit** (material property), an extra repulsive term is added so particles do not pass through each other:  
  **F** += (compression_limit − r) · k · 3 · **ê**

### Time integration

- Fixed base step with **4 substeps** per frame.
- Per substep: collision handling → spring forces (and bond removal) → **v** += **F**/m · dt → **x** += **v** · dt.
- Static (boundary) particles have **v** = 0 and are not updated.

### Damping

- Applied only to **armor** particles that still belong to the **connected component of the fixed boundary** (flood-fill from static particles along springs).
- Those particles: **v** *= 0.98 each substep.
- Detached fragments and the penetrator are not damped.

### Collisions

**Broad phase:** spatial hash with cell size ≈ 1.5 × collision distance; only particles in the same or neighboring cells are tested.

**Pair test:** two particles collide if (i) distance < collision distance, or (ii) relative velocity is large and they would reach that distance within the substep (predicted distance).

**Resolution:**

- Overlap **δ** = collision_distance − distance; **n** = unit vector between particles.
- Overlap is corrected by moving each particle along **n**: split by strength ratio  
  **ratio₁** = s₂/(s₁+s₂), **ratio₂** = s₁/(s₁+s₂), scaled by a contact stiffness factor from (s₁+s₂) and a correction factor (0.8 penetrator–armor, 0.5 same-body).

**Impulse (approaching pair, v_rel · n > 0):**

- Reduced mass μ = m₁m₂/(m₁+m₂).
- Restitution **e** from material **e₁,e₂**: base **e** = √(e₁e₂), reduced at high impact speed; strength ratio **s_ratio** = min(s₁,s₂)/max(s₁,s₂) adds loss term (1 − s_ratio)·0.4 (capped).
- Impulse magnitude: **J** = (1 + e)·μ·(v_rel·n), clamped; **Δv₁** = −**J n**/m₁, **Δv₂** = **J n**/m₂.

**Separating or slow contact:** repulsive force from overlap and stiffness (same-body uses average Young’s modulus and contact area; cross-body uses overlap-based force), converted to impulse and capped.

---

## Materials

Per-material constants (from real-world data where noted):

| Property | Role |
|----------|------|
| **density** | Mass = density × spacing³ |
| **youngs_modulus** | Spring stiffness **k** |
| **strength** | Overlap split (ratio), restitution loss (s_ratio), contact stiffness factor |
| **restitution** | Impact bounce (combined √(e₁e₂)) |
| **hardness** | Stored for possible future use |
| **poisson_ratio** | Stored for possible future use |
| **friction_coeff** | Used in restitution loss at high speed |
| **breaking_strain** | |ε| > this → spring removed |
| **compression_limit** | L/L₀ below this → extra repulsion |

Representative values: **Tungsten** (penetrator) – ρ≈19.3e3 kg/m³, E≈411 GPa, breaking_strain 0.05; **RHA Steel** – ρ≈7850, E≈210 GPa, strength≈1 GPa, breaking_strain 0.15; **Ceramic** – ρ≈3950, E≈370 GPa, breaking_strain 0.008. References: AZoM, material-properties.org, MIL-DTL-12560K, ceramic armor literature.

---

## Numerical details

- **Substeps:** 4 per frame; time scale slider multiplies effective dt.
- **Collision distance:** fixed (e.g. 0.012 m); particle radius used for overlap and response.
- **Activation:** penetrator always “active”; armor particles become active when velocity² > threshold or via propagation along springs from collisions.
- **Angular velocity:** computed from spring torques about center of mass (for visualization / possible future use); primary motion is still linear **v** from spring forces.

---

## Project layout

```
├── armorsim_qt.py   # Application, physics, and rendering
├── requirements.txt
└── README.md
```
