# -*- coding: utf-8 -*-
"""Material properties: density, strength, Young's modulus, breaking_strain, compression_limit, etc."""

import numpy as np

MATERIALS = {
    'tungsten': {
        'name': 'Tungsten (pure)',
        'role': 'penetrator',
        'density': 19300,
        'strength': 980e6,
        'youngs_modulus': 411e9,
        'restitution': 0.16,
        'hardness': 7.5,
        'poisson_ratio': 0.28,
        'friction_coeff': 0.4,
        'breaking_strain': 0.012,  # polycrystalline W is brittle: 1-2% elongation at break
        'yield_strain': 0.002,     # ~750 MPa yield / 411 GPa = 0.0018; onset of micro-plasticity ~0.002
        'compression_limit': 0.40,
        'spring_min_ratio': 0.95,  # per-spring limit; compounds across chain (~5 springs → 0.95^5≈0.77 global)
        'shear_boost': 2.3,        # G = E/(2(1+ν)) = 411/(2×1.28) = 161 GPa → G/E = 0.391
        'color': np.array([1.0, 0.7, 0.25])
    },
    'tungsten_carbide': {
        'name': 'Tungsten Carbide',
        'role': 'penetrator',
        'density': 15000,          # WC-6Co: 15.0-15.1 g/cm³ (General Carbide, AZoM)
        'strength': 2700e6,        # WC-Co compressive strength ~2.7 GPa; TRS ~2.0-2.5 GPa
        'youngs_modulus': 630e9,   # WC-6Co: 614-648 GPa range (Goodfellow, AZoM)
        'restitution': 0.12,
        'hardness': 9.5,
        'poisson_ratio': 0.22,
        'friction_coeff': 0.35,
        'breaking_strain': 0.004,  # ~0.4% fracture strain in tension (very brittle cermet)
        'yield_strain': 0.003,     # onset of micro-cracking before fracture
        'compression_limit': 0.55,
        'spring_min_ratio': 0.97,  # WC very stiff/brittle; fractures before compressing; ~0.97^5≈0.86 global
        'shear_boost': 2.4,        # G = E/(2(1+ν)) = 630/(2×1.22) = 258 GPa → G/E = 0.410 (highest of the group)
        'color': np.array([0.85, 0.78, 0.65])
    },
    'depleted_uranium': {
        'name': 'Depleted Uranium',
        'role': 'penetrator',
        'density': 19050,          # U-0.75Ti: 19.0-19.05 g/cm³ (OSTI reports)
        'strength': 1200e6,        # U-0.75Ti aged condition UTS ~1200 MPa (OSTI 6998138)
        'youngs_modulus': 175e9,   # Kessler 1987 J.Nucl.Mater: E = 175-193 GPa aged alpha-prime
        'restitution': 0.15,
        'hardness': 6.5,           # HB ~270-350; self-sharpening adiabatic shear behaviour
        'poisson_ratio': 0.23,     # Kessler 1987: ~0.23 for standard aged microstructure
        'friction_coeff': 0.45,
        'breaking_strain': 0.08,   # ductility maintained: elongation 5-12%; use 0.08
        'yield_strain': 0.006,     # ~1000 MPa yield / 175 GPa = 0.00571
        'compression_limit': 0.45,
        'spring_min_ratio': 0.90,  # DU more ductile; ~0.90^5≈0.59 global (allows meaningful nose crush)
        'shear_boost': 2.4,        # G = E/(2(1+ν)) = 175/(2×1.23) = 71 GPa → G/E = 0.406; ductile but shear ratio similar to W
        'color': np.array([0.6, 0.62, 0.5])
    },
    'steel_core': {
        'name': 'Steel (AP core)',
        'role': 'penetrator',
        'density': 7850,
        'strength': 2100e6,        # 60 HRC SAE 52100: UTS ~2100-2150 MPa
        'youngs_modulus': 210e9,
        'restitution': 0.26,
        'hardness': 7.0,           # 60 HRC ≈ HV~746; well above RHA (~360 HB)
        'poisson_ratio': 0.29,
        'friction_coeff': 0.55,
        'breaking_strain': 0.06,   # at 60 HRC elongation ~6-8% (embrittled by hardening)
        'yield_strain': 0.0086,    # ~1800 MPa yield / 210 GPa = 0.00857
        'compression_limit': 0.48,
        'spring_min_ratio': 0.93,  # hardened steel; ~0.93^5≈0.70 global; stiff enough to limit sponging
        'shear_boost': 2.2,        # G = E/(2(1+ν)) = 210/(2×1.29) = 81 GPa → G/E = 0.387
        'color': np.array([0.55, 0.52, 0.58])
    },
    'steel': {
        'name': 'RHA Steel',
        'role': 'both',
        'density': 7850,
        'strength': 1207e6,        # MIL-A-12560 Class 3: 175 ksi = 1207 MPa
        'youngs_modulus': 207e9,   # Q&T low-alloy steel; modulus barely changes with heat treatment
        'restitution': 0.30,
        'hardness': 5.5,           # HB 302-360 for Class 3 RHA
        'poisson_ratio': 0.29,
        'friction_coeff': 0.6,
        'breaking_strain': 0.14,   # MIL-A-12560 spec: 12% elongation in 2 inches ~0.14
        'yield_strain': 0.0053,    # ~1090 MPa yield / 207 GPa = 0.00527
        'compression_limit': 0.50,
        'spring_min_ratio': 0.88,  # more ductile than AP core; ~0.88^5≈0.53 global when used as penetrator
        'shear_boost': 2.2,        # G = E/(2(1+ν)) = 207/(2×1.29) = 80 GPa → G/E = 0.387
        'color': np.array([0.5, 0.55, 0.65])
    },
    'ceramic': {
        'name': 'Alumina (Al2O3)',
        'role': 'armor',
        'density': 3960,           # 99% purity: 3.96 g/cm³ (AZoM 99.5%)
        'strength': 380e6,         # MOR flexural 350-400 MPa; compressive >2 GPa
        'youngs_modulus': 370e9,   # 99.5% Al2O3: MatWeb, AZoM confirm ~370 GPa
        'restitution': 0.13,
        'hardness': 9.0,           # HV 1400-1600; anchor for ceramics
        'poisson_ratio': 0.22,
        'friction_coeff': 0.5,
        'breaking_strain': 0.001,  # brittle: fracture at elastic limit; 380 MPa / 370 GPa = 0.00103
        'yield_strain': 0.001,     # no plastic yield zone — fracture = yield for ceramics
        'compression_limit': 0.65,
        'spring_min_ratio': 0.0,   # armor only; clamp not applied to penetrators
        'color': np.array([0.95, 0.9, 0.82])
    },
    'boron_carbide': {
        'name': 'Boron Carbide (B4C)',
        'role': 'armor',
        'density': 2520,
        'strength': 585e6,         # 4-point bending: 585 ± 70 MPa (Vargas-Gonzalez 2010, IJACT)
        'youngs_modulus': 460e9,   # RUS measurement 458.7 GPa; AZoM/MakeItFrom confirm ~460 GPa
        'restitution': 0.10,
        'hardness': 9.8,           # HV ~38 GPa; 3rd hardest material (after diamond, cBN)
        'poisson_ratio': 0.17,     # 0.172 (AZoM, B4C Kessler study)
        'friction_coeff': 0.45,
        'breaking_strain': 0.00127, # 585 MPa / 460 GPa = 0.00127; fully brittle
        'yield_strain': 0.00127,    # no plastic zone; fracture = yield
        'compression_limit': 0.70,
        'spring_min_ratio': 0.0,    # armor only
        'color': np.array([0.72, 0.72, 0.72])
    },
    'silicon_carbide': {
        'name': 'Silicon Carbide (SiC)',
        'role': 'armor',
        'density': 3210,
        'strength': 400e6,         # SiC flexural 360-520 MPa; armor-grade pressureless sintered ~400 MPa
        'youngs_modulus': 410e9,
        'restitution': 0.12,
        'hardness': 9.3,           # HV ~2500; between Al2O3 and B4C
        'poisson_ratio': 0.19,
        'friction_coeff': 0.48,
        'breaking_strain': 0.00088, # brittle: ~400 MPa / 410 GPa = 0.000976; ~0.001
        'yield_strain': 0.00088,    # no plastic zone
        'compression_limit': 0.68,
        'spring_min_ratio': 0.0,    # armor only
        'color': np.array([0.65, 0.68, 0.66])
    },
    'titanium': {
        'name': 'Titanium (Ti-6Al-4V)',
        'role': 'armor',
        'density': 4430,
        'strength': 1170e6,        # Ti-6Al-4V STA: UTS 1170 MPa (MatWeb ASM)
        'youngs_modulus': 114e9,   # 113-116 GPa across all sources
        'restitution': 0.28,
        'hardness': 4.8,           # HB 334-379; clearly below RHA (5.5) — fixes titanium < RHA
        'poisson_ratio': 0.34,
        'friction_coeff': 0.52,    # Ti-6Al-4V has notably high friction (~0.5-0.6) due to galling
        'breaking_strain': 0.10,   # STA elongation 10-14%; conservative 10%
        'yield_strain': 0.0096,    # ~1100 MPa yield / 114 GPa = 0.00965
        'compression_limit': 0.48,
        'spring_min_ratio': 0.0,   # armor only
        'color': np.array([0.7, 0.7, 0.72])
    },
    'aluminum_armor': {
        'name': 'Aluminum 5083',
        'role': 'armor',
        'density': 2660,           # 5083: 2.66 g/cm³ ✓
        'strength': 290e6,         # 5083-H116: UTS ~290 MPa
        'youngs_modulus': 70.3e9,  # 5083: 70.3 GPa
        'restitution': 0.30,
        'hardness': 2.8,           # HB ~80 (5083); below mild steel
        'poisson_ratio': 0.33,
        'friction_coeff': 0.55,
        'breaking_strain': 0.16,   # 5083-H116 elongation ~16%
        'yield_strain': 0.003,     # ~215 MPa yield / 70.3 GPa = 0.00306
        'compression_limit': 0.50,
        'spring_min_ratio': 0.0,   # armor only
        'color': np.array([0.82, 0.84, 0.86])
    },
    'uhmwpe': {
        'name': 'UHMWPE (Dyneema)',
        'role': 'armor',
        'density': 970,            # UHMWPE fiber: 0.97 g/cm³; panel ~970-980 kg/m³
        'strength': 3.4e9,         # Dyneema SK76 fiber ≥3.0-3.4 GPa; composite ~3.4 GPa longitudinal
        'youngs_modulus': 100e9,   # fiber modulus 90-120 GPa; composite in-plane ~100 GPa
        'restitution': 0.20,
        'hardness': 1.8,           # Shore D ~60-65; very soft vs metals
        'poisson_ratio': 0.42,
        'friction_coeff': 0.25,    # UHMWPE: one of the lowest dry friction coefficients of any solid
        'breaking_strain': 0.035,  # Dyneema SK76: elongation at break ~3.5%
        'yield_strain': 0.015,     # ~1.5% onset of fibre/matrix damage
        'compression_limit': 0.35,
        'spring_min_ratio': 0.0,   # armor only
        'color': np.array([0.92, 0.93, 0.95])
    },
    'adamantium': {
        'name': 'Adamantium',
        'role': 'both',
        'density': 5e8,
        'strength': 2e12,
        'youngs_modulus': 2000e9,
        'restitution': 0.0,
        'hardness': 10.0,
        'poisson_ratio': 0.20,
        'friction_coeff': 0.15,
        'breaking_strain': 1.0,
        'yield_strain': 0.02,
        'compression_limit': 0.995,
        'spring_min_ratio': 0.99,  # effectively indestructible; springs cannot compress
        'shear_boost': 5.0,        # effectively indestructible; no lateral deformation
        'color': np.array([0.75, 0.78, 0.85])
    },
}
