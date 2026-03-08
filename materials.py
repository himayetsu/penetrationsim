# -*- coding: utf-8 -*-
"""Material properties: density, strength, Young's modulus, breaking_strain, compression_limit, etc."""

import numpy as np

# Real-world data refs: tungsten (AZoM, material-properties.org);
# RHA (MIL-DTL-12560K); ceramic armor (alumina/B4C, AZoM, precision-ceramics).
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
