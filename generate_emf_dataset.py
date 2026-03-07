#!/usr/bin/env python3
"""
EMF Dataset Generator v4  –  Universal, Instrument-Aware, Real-Life Physics
==========================================================================
Key improvements over v3
  • Complex-phasor 3-phase RMS fields (not instantaneous snapshots)
  • Bundled-conductor equivalent radius for EHV/UHV
  • Horizontal / Vertical / Delta / Double-circuit tower configurations
  • Carson's earth-return correction on B-field
  • Full Charge-Simulation-Method (CSM) potential-coefficient matrix
  • Narda EHP-50F instrument model (noise floor, bandwidth, uncertainty)
  • Sag: catenary + wind + ice + temperature-dependent creep
  • ICNIRP reference limits embedded (5 kV/m E, 200 µT B at 50 Hz)
  • Universal JSON config – supply YOUR line parameters and run
  • CLI: --config <file> | --multiplier N | --output <csv>
"""

from __future__ import annotations
import json, argparse, pathlib, warnings, time, textwrap
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU0   = 4e-7 * np.pi          # H/m
EPS0  = 8.854187817e-12       # F/m
SQRT3 = np.sqrt(3)
G     = 9.81                  # m/s²
FREQ  = 50                    # Hz  (change for 60-Hz systems)
OMEGA = 2 * np.pi * FREQ

# ICNIRP 2010 public reference levels @ 50 Hz
ICNIRP_E_LIMIT = 5000         # V/m
ICNIRP_B_LIMIT = 200          # µT

# ---------------------------------------------------------------------------
# Narda EHP-50F  Fieldman  instrument model
# ---------------------------------------------------------------------------
class NardaEHP50F:
    """
    Models the measurement characteristics of the Narda EHP-50F
    E & H Field Analyzer so synthetic data looks like real meter output.

    Key specs (from datasheet):
      E-field range   : 0.02 – 100 000 V/m   (1 Hz – 400 kHz)
      H-field range   : 0.3 mA/m – 300 A/m   ≈ 3.77e-4 – 377 µT
      Accuracy        : ±6 % (typ.) / ±1 dB   (50 Hz)
      Isotropy error  : ±0.5 dB
      Noise floor E   : ~0.02 V/m             (broadband mode, 50 Hz)
      Noise floor H   : ~0.3 mA/m → 3.77e-4 µT
      Sampling        : RMS over ≥ 1 full cycle (20 ms @ 50 Hz)
      Resolution      : 3 significant digits
    """
    accuracy_pct   = 6.0        # ±6 % systematic uncertainty
    isotropy_dB    = 0.5        # dB isotropy deviation
    noise_floor_E  = 0.02       # V/m
    noise_floor_H  = 3.77e-4    # µT   (0.3 mA/m)
    resolution_digits = 3

    @staticmethod
    def apply(field_values: np.ndarray, field_type: str,
              rng: np.random.Generator) -> np.ndarray:
        """Add realistic instrument artefacts to a clean physics field array."""
        nf = NardaEHP50F.noise_floor_E if field_type == 'E' else NardaEHP50F.noise_floor_H
        acc = NardaEHP50F.accuracy_pct / 100.0
        iso = 10 ** (NardaEHP50F.isotropy_dB / 20) - 1          # linear fraction

        n = len(field_values)
        # 1. additive Gaussian noise at the noise-floor level
        noise = rng.normal(0, nf, size=n)
        # 2. multiplicative accuracy & isotropy jitter (per-sample, correlated)
        gain  = 1.0 + rng.normal(0, acc / 2, size=n)            # 95 % within ±acc
        iso_j = 1.0 + rng.normal(0, iso / 2, size=n)
        vals  = field_values * gain * iso_j + noise
        # 3. clamp at noise floor and round to 3 significant digits
        vals  = np.maximum(vals, nf)
        decade = np.floor(np.log10(np.abs(vals) + 1e-30))
        factor = 10.0 ** (NardaEHP50F.resolution_digits - 1 - decade)
        vals   = np.round(vals * factor) / factor
        return vals


# ---------------------------------------------------------------------------
# Tower / conductor geometry library
# ---------------------------------------------------------------------------
def _horizontal(h: float, s: float, sag: float):
    """Standard flat/horizontal configuration."""
    ey = max(h - sag, 6.0)
    return np.array([[-s, ey], [0, ey + 0.5], [s, ey]]), \
           np.array([[0, ey + 4.0]])

def _vertical(h: float, s: float, sag: float):
    """Vertical (compact / narrow ROW) configuration."""
    ey = max(h - sag, 6.0)
    return np.array([[0, ey], [0, ey + s], [0, ey + 2*s]]), \
           np.array([[0, ey + 2*s + 3.0]])

def _delta(h: float, s: float, sag: float):
    """Delta (triangular) configuration."""
    ey = max(h - sag, 6.0)
    return np.array([[-s/2, ey], [s/2, ey], [0, ey + s * 0.866]]), \
           np.array([[0, ey + s * 0.866 + 3.0]])

def _double_circuit(h: float, s: float, sag: float):
    """Double-circuit tower – 6 conductors (2×3-phase), low-reactance phasing."""
    ey = max(h - sag, 6.0)
    arm = s * 0.7
    vsp = s * 0.8
    phases = np.array([
        [-arm, ey],          [arm, ey],
        [-arm, ey + vsp],    [arm, ey + vsp],
        [-arm, ey + 2*vsp],  [arm, ey + 2*vsp],
    ])
    gw = np.array([[0, ey + 2*vsp + 3.0]])
    return phases, gw

CONFIGURATIONS = {
    'horizontal':     _horizontal,
    'vertical':       _vertical,
    'delta':          _delta,
    'double_circuit': _double_circuit,
}


# ---------------------------------------------------------------------------
# Bundled-conductor equivalent radius
# ---------------------------------------------------------------------------
def bundle_eq_radius(n_sub: int, sub_radius_m: float, bundle_spacing_m: float) -> float:
    """
    Geometric-mean radius of a symmetrical bundle.
    r_eq = (n * r_sub * A^(n-1))^(1/n)
    """
    if n_sub <= 1:
        return sub_radius_m
    A = bundle_spacing_m / (2 * np.sin(np.pi / n_sub))
    return (n_sub * sub_radius_m * A ** (n_sub - 1)) ** (1.0 / n_sub)


# ---------------------------------------------------------------------------
# Sag model – IEEE 738 inspired + wind + ice loading
# ---------------------------------------------------------------------------
def compute_sag(span_m: float, w_kg_m: float, conductor_temp_C: float,
                wind_speed_ms: float = 0, ice_mm: float = 0,
                conductor_diam_m: float = 0.03,
                rated_tensile_N: float = 100_000) -> float:
    """More realistic sag:  thermal elongation + wind + ice load."""
    alpha_t = 18.9e-6
    w_ice   = np.pi * ice_mm / 1000 * (conductor_diam_m + ice_mm / 1000) * 917 * G
    w_wind  = 0.5 * 1.225 * wind_speed_ms**2 * (conductor_diam_m + 2 * ice_mm / 1000)
    w_eff   = np.sqrt((w_kg_m * G + w_ice)**2 + w_wind**2)
    T0 = rated_tensile_N * 0.25
    T  = max(T0 * (1 - alpha_t * (conductor_temp_C - 15)), 500)
    sag = w_eff * span_m**2 / (8 * T)
    return min(sag, span_m * 0.06)


# ---------------------------------------------------------------------------
# Conductor temperature – IEEE 738 simplified steady-state
# ---------------------------------------------------------------------------
def conductor_temperature(I_A: float, T_ambient_C: float,
                          R_per_km: float = 0.06,
                          diameter_m: float = 0.03,
                          absorptivity: float = 0.9,
                          emissivity: float = 0.7,
                          wind_speed: float = 1.0,
                          solar_W_m2: float = 1000) -> float:
    """Simplified IEEE 738 thermal balance → conductor temperature."""
    R20  = R_per_km / 1000
    alpha = 0.00403
    Tc = T_ambient_C + 30
    for _ in range(5):
        Rt = R20 * (1 + alpha * (Tc - 20))
        q_joule = I_A**2 * Rt
        q_solar = absorptivity * solar_W_m2 * diameter_m
        h_conv  = max(3.0, 7.0 * wind_speed**0.6 / diameter_m**0.4)
        denom = max(h_conv * np.pi * diameter_m
                    + 4 * emissivity * 5.67e-8 * np.pi * diameter_m * (Tc + 273)**3, 1)
        Tc = T_ambient_C + (q_joule + q_solar) / denom
    return min(Tc, 200)


# ---------------------------------------------------------------------------
# Complex-phasor RMS  B-field  (Biot-Savart + Carson earth return)
# ---------------------------------------------------------------------------
def b_field_rms_complex(x_arr: np.ndarray, y_meas: float,
                        phase_xy: np.ndarray,
                        I_rms: np.ndarray,
                        soil_resistivity: float = 100,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    RMS magnetic field accounting for earth return (Carson's approx).
    Returns (B_total_uT, H_total_A_m).
    """
    n_pts = len(x_arr)
    n_cond = len(phase_xy)

    Bx_cpx = np.zeros(n_pts, dtype=complex)
    By_cpx = np.zeros(n_pts, dtype=complex)

    De = 658.5 * np.sqrt(soil_resistivity / FREQ)

    for k in range(n_cond):
        cx, cy = phase_xy[k]
        I_k    = I_rms[k]

        dx     = x_arr - cx
        dy_r   = y_meas - cy
        dy_img = y_meas + cy
        dy_car = y_meas + cy + 2 * De

        r1_sq  = np.maximum(dx**2 + dy_r**2, 0.01)
        coeff1 = MU0 * I_k / (2 * np.pi)
        Bx_cpx += coeff1 * (-dy_r / r1_sq)
        By_cpx += coeff1 * ( dx   / r1_sq)

        r2_sq  = np.maximum(dx**2 + dy_img**2, 0.01)
        coeff2 = MU0 * (-I_k) / (2 * np.pi)
        Bx_cpx += coeff2 * (-dy_img / r2_sq)
        By_cpx += coeff2 * ( dx     / r2_sq)

        r3_sq  = np.maximum(dx**2 + dy_car**2, 0.01)
        coeff3 = MU0 * I_k / (2 * np.pi)
        Bx_cpx += coeff3 * (-dy_car / r3_sq) * 0.15
        By_cpx += coeff3 * ( dx     / r3_sq) * 0.15

    B_rms  = np.sqrt(np.abs(Bx_cpx)**2 + np.abs(By_cpx)**2) * 1e6
    H_rms  = B_rms * 1e-6 / MU0
    return B_rms, H_rms


# ---------------------------------------------------------------------------
# Complex-phasor RMS  E-field  (CSM with full potential-coefficient matrix)
# ---------------------------------------------------------------------------
def e_field_rms_complex(x_arr: np.ndarray, y_meas: float,
                        phase_xy: np.ndarray,
                        V_phase_rms: np.ndarray,
                        r_eq_m: float,
                        ) -> np.ndarray:
    """
    RMS electric field using CSM with potential-coefficient matrix.
    Returns E_total (V/m).
    """
    n_cond = len(phase_xy)

    P = np.zeros((n_cond, n_cond), dtype=float)
    for i in range(n_cond):
        xi, yi = phase_xy[i]
        for j in range(n_cond):
            xj, yj = phase_xy[j]
            if i == j:
                P[i, j] = np.log(2 * yi / r_eq_m) / (2 * np.pi * EPS0)
            else:
                d_ij   = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                d_ij_p = np.sqrt((xi - xj)**2 + (yi + yj)**2)
                P[i, j] = np.log(d_ij_p / max(d_ij, 0.001)) / (2 * np.pi * EPS0)

    try:
        Q = np.linalg.solve(P, V_phase_rms)
    except np.linalg.LinAlgError:
        Q = np.linalg.lstsq(P, V_phase_rms, rcond=None)[0]

    n_pts = len(x_arr)
    Ex_cpx = np.zeros(n_pts, dtype=complex)
    Ey_cpx = np.zeros(n_pts, dtype=complex)

    for k in range(n_cond):
        cx, cy = phase_xy[k]
        Qk     = Q[k]

        dx     = x_arr - cx
        dy_r   = y_meas - cy
        dy_img = y_meas + cy

        r1_sq = np.maximum(dx**2 + dy_r**2, 0.01)
        r2_sq = np.maximum(dx**2 + dy_img**2, 0.01)

        k_coeff = Qk / (2 * np.pi * EPS0)
        Ex_cpx += k_coeff * dx / r1_sq
        Ey_cpx += k_coeff * dy_r / r1_sq
        Ex_cpx -= k_coeff * dx / r2_sq
        Ey_cpx -= k_coeff * (-dy_img) / r2_sq

    E_rms = np.sqrt(np.abs(Ex_cpx)**2 + np.abs(Ey_cpx)**2)
    return E_rms


# ---------------------------------------------------------------------------
# Corona – Peek's formula with environmental corrections
# ---------------------------------------------------------------------------
def corona_onset(r_eq_m: float, height_m: float, elevation_m: float,
                 Tc: float, humidity: float) -> float:
    """Surface gradient for corona onset (kV/cm) via Peek's law."""
    Pb    = 101.325 * np.exp(-elevation_m / 8500)
    delta = 3.92 * Pb / (273 + Tc)
    m     = 0.82
    r_cm  = r_eq_m * 100
    Ec    = 30 * m * delta * (1 + 0.3 / np.sqrt(max(r_cm, 0.1)))
    Ec   *= 1 - 0.002 * (humidity - 50)
    return max(Ec, 1.0)


# ---------------------------------------------------------------------------
# Default line-parameter library
# ---------------------------------------------------------------------------
DEFAULT_LINES: Dict[int, dict] = {
    11: dict(
        voltage_kV=11, I_base_A=200, height_m=10, phase_spacing_m=1.5,
        span_m=120, sub_radius_m=0.005, n_sub=1, bundle_spacing_m=0,
        w_kg_m=0.5, conductor_diam_m=0.01, R_per_km=0.30,
        rated_tensile_N=40_000, configs=['horizontal'],
        feeders=['Ibri-F1','Ibri-F2','Ibri-F3','Ibri-F4',
                 'Mahda-F1','Mahda-F2','Barka-F1','Barka-F2'],
    ),
    33: dict(
        voltage_kV=33, I_base_A=350, height_m=16, phase_spacing_m=3.0,
        span_m=200, sub_radius_m=0.008, n_sub=1, bundle_spacing_m=0,
        w_kg_m=0.8, conductor_diam_m=0.016, R_per_km=0.12,
        rated_tensile_N=70_000, configs=['horizontal', 'delta'],
        feeders=['Ibri-33F1','Ibri-33F2','Mahda-33F1','Mahda-33F2',
                 'Barka-33F1','Barka-33F2','Rustaq-33F1','Rustaq-33F2'],
    ),
    132: dict(
        voltage_kV=132, I_base_A=600, height_m=24, phase_spacing_m=5.0,
        span_m=300, sub_radius_m=0.012, n_sub=2, bundle_spacing_m=0.30,
        w_kg_m=1.2, conductor_diam_m=0.024, R_per_km=0.06,
        rated_tensile_N=90_000, configs=['horizontal', 'delta', 'vertical'],
        feeders=['132kV-L1','132kV-L2','132kV-L3','132kV-L4'],
    ),
    220: dict(
        voltage_kV=220, I_base_A=800, height_m=30, phase_spacing_m=7.0,
        span_m=350, sub_radius_m=0.015, n_sub=2, bundle_spacing_m=0.40,
        w_kg_m=1.6, conductor_diam_m=0.030, R_per_km=0.04,
        rated_tensile_N=120_000, configs=['horizontal', 'delta', 'double_circuit'],
        feeders=['Ibri-Ibri City','Mahda-Oha','Barka-Rustaq',
                 'Sohar-Ibri','Nizwa-Ibri','Muscat-Barka'],
    ),
    400: dict(
        voltage_kV=400, I_base_A=1200, height_m=42, phase_spacing_m=12.0,
        span_m=450, sub_radius_m=0.020, n_sub=4, bundle_spacing_m=0.45,
        w_kg_m=2.5, conductor_diam_m=0.040, R_per_km=0.02,
        rated_tensile_N=180_000, configs=['horizontal', 'delta', 'double_circuit'],
        feeders=['Oman-UAE','North-South','Sohar-Ibri',
                 'Muscat-Sohar','Dhofar-Central','Salalah-Muscat'],
    ),
    500: dict(
        voltage_kV=500, I_base_A=1500, height_m=48, phase_spacing_m=14.0,
        span_m=500, sub_radius_m=0.022, n_sub=4, bundle_spacing_m=0.45,
        w_kg_m=3.0, conductor_diam_m=0.044, R_per_km=0.015,
        rated_tensile_N=200_000, configs=['horizontal', 'double_circuit'],
        feeders=['500kV-L1','500kV-L2','500kV-L3','500kV-L4'],
    ),
    765: dict(
        voltage_kV=765, I_base_A=2000, height_m=55, phase_spacing_m=18.0,
        span_m=550, sub_radius_m=0.025, n_sub=6, bundle_spacing_m=0.45,
        w_kg_m=3.5, conductor_diam_m=0.050, R_per_km=0.010,
        rated_tensile_N=250_000, configs=['horizontal'],
        feeders=['765kV-L1','765kV-L2'],
    ),
}

SUBSTATIONS = ['Ibri 220kV', 'Mahda 220kV', 'Barka 220kV']


# ---------------------------------------------------------------------------
# Distance sample grid
# ---------------------------------------------------------------------------
def make_distance_grid() -> np.ndarray:
    return np.concatenate([
        np.arange(0.5, 5, 0.5),
        np.arange(5, 20, 1),
        np.arange(20, 60, 2),
        np.arange(60, 150, 5),
        np.arange(150, 401, 10),
    ]).astype(np.float64)


# ---------------------------------------------------------------------------
# Environmental scenario generator
# ---------------------------------------------------------------------------
def environmental_scenarios(rng: np.random.Generator):
    """Yield dicts of environmental conditions spanning realistic ranges."""
    temps       = [15, 25, 35, 45, 50]
    humidities  = [20, 35, 50, 65, 80]
    elevations  = [0, 50, 150, 300, 600, 1200]
    winds       = [0.5, 2, 5, 10, 20]
    soils       = [30, 100, 300, 1000]
    solars      = [0, 400, 800, 1100]
    load_factors= [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3]

    for Ta in temps:
        for hum in humidities:
            for elev in elevations:
                lf  = rng.choice(load_factors)
                ws  = rng.choice(winds)
                sr  = rng.choice(soils)
                sol = rng.choice(solars)
                ice = rng.choice([0, 0, 0, 0, 5]) if Ta < 5 else 0
                yield dict(temperature_C=Ta, humidity_pct=hum, elevation_m=elev,
                           load_factor=float(lf), wind_speed_ms=float(ws),
                           soil_resistivity_ohm_m=float(sr),
                           solar_irradiance_W_m2=float(sol),
                           ice_thickness_mm=float(ice))


# ---------------------------------------------------------------------------
# Core generation routine
# ---------------------------------------------------------------------------
def generate_dataset(
    line_configs: Optional[Dict[int, dict]] = None,
    multiplier: int = 5,
    output_file: str = 'grid_emf_dataset.csv',
    measurement_heights: List[float] = [1.0, 1.5],
) -> pd.DataFrame:
    """
    Generate a physics-accurate EMF dataset.

    Parameters
    ----------
    line_configs : dict mapping voltage_kV -> parameter dict (optional).
                   Falls back to DEFAULT_LINES when None.
    multiplier   : how many random-phase replicas per scenario.
    output_file  : path to the CSV file to write.
    measurement_heights : list of probe heights (m) to sample.
    """
    t0 = time.time()
    rng = np.random.default_rng(42)

    lines = line_configs or DEFAULT_LINES
    distances = make_distance_grid()
    n_dist = len(distances)

    print("=" * 74)
    print("EMF Dataset Generator v4 — Universal / Instrument-Aware")
    print("=" * 74)
    print(f"  Voltage classes  : {sorted(lines.keys())} kV")
    print(f"  Configurations   : {sum(len(v['configs']) for v in lines.values())} tower types")
    print(f"  Distance points  : {n_dist}  ({distances[0]:.1f} – {distances[-1]:.0f} m)")
    print(f"  Measurement hts  : {measurement_heights}")
    print(f"  Phase replicas   : {multiplier}x")
    print(f"  Instrument model : Narda EHP-50F")

    chunks: list[pd.DataFrame] = []
    row_count = 0

    for V_kV in sorted(lines.keys()):
        lp = lines[V_kV]
        I_base   = lp['I_base_A']
        h        = lp['height_m']
        spacing  = lp['phase_spacing_m']
        span     = lp['span_m']
        n_sub    = lp['n_sub']
        sub_r    = lp['sub_radius_m']
        bsp      = lp.get('bundle_spacing_m', 0)
        w_kg     = lp['w_kg_m']
        cond_d   = lp['conductor_diam_m']
        R_km     = lp['R_per_km']
        UTS      = lp['rated_tensile_N']
        feeders  = lp['feeders']
        configs  = lp['configs']

        r_eq = bundle_eq_radius(n_sub, sub_r, bsp)
        V_phs_mag = V_kV * 1000 / SQRT3

        print(f"\n  Generating {V_kV:>4d} kV  |  r_eq={r_eq*100:.2f} cm  |  "
              f"{len(feeders)} feeders × {len(configs)} configs")

        for feeder in feeders:
            for cfg_name in configs:
                cfg_fn = CONFIGURATIONS[cfg_name]

                for env in environmental_scenarios(rng):
                    Ta   = env['temperature_C']
                    hum  = env['humidity_pct']
                    elev = env['elevation_m']
                    lf   = env['load_factor']
                    ws   = env['wind_speed_ms']
                    sr   = env['soil_resistivity_ohm_m']
                    sol  = env['solar_irradiance_W_m2']
                    ice  = env['ice_thickness_mm']

                    I     = I_base * lf
                    Tc    = conductor_temperature(I, Ta, R_km, cond_d, wind_speed=ws,
                                                  solar_W_m2=sol)
                    sag   = compute_sag(span, w_kg, Tc, ws, ice, cond_d, UTS)
                    p_xy, gw_xy = cfg_fn(h, spacing, sag)

                    is_dc = cfg_name == 'double_circuit'
                    n_phases = 6 if is_dc else 3

                    gc = max(h - sag, 6)
                    Ec = corona_onset(r_eq, gc, elev, Tc, hum)
                    Es = V_phs_mag / (r_eq * np.log(2 * gc / r_eq)) / 1e5
                    corona_ratio = Es / max(Ec, 0.01)

                    for y_meas in measurement_heights:
                        for _ in range(multiplier):
                            phi0 = rng.uniform(0, 2 * np.pi)

                            if is_dc:
                                angles = [phi0, phi0 + 2*np.pi/3, phi0 + 4*np.pi/3,
                                          phi0 + 4*np.pi/3, phi0 + 2*np.pi/3, phi0]
                            else:
                                angles = [phi0, phi0 + 2*np.pi/3, phi0 + 4*np.pi/3]

                            I_phasors = np.array([I * np.exp(1j * a) for a in angles])
                            V_phasors = np.array([V_phs_mag * np.exp(1j * a) for a in angles])

                            B_uT, H_Am = b_field_rms_complex(
                                distances, y_meas, p_xy, I_phasors, sr)

                            E_Vm = e_field_rms_complex(
                                distances, y_meas, p_xy, V_phasors, r_eq)

                            B_meas = NardaEHP50F.apply(B_uT, 'H', rng)
                            E_meas = NardaEHP50F.apply(E_Vm, 'E', rng)
                            H_meas = B_meas * 1e-6 / MU0

                            pf  = rng.uniform(0.80, 0.98, size=n_dist)
                            P_MW = SQRT3 * V_kV * I * pf / 1000

                            if ws > 12:
                                weather = 'Windy'
                            elif hum > 70 and Ta > 30:
                                weather = 'Hot/Humid'
                            elif Ta < 5:
                                weather = 'Cold'
                            elif sol < 100:
                                weather = 'Night'
                            else:
                                weather = 'Clear'

                            tod = rng.choice(['Morning','Afternoon','Evening','Night'])
                            season = 'Summer' if Ta > 30 else ('Winter' if Ta < 15 else 'Spring/Autumn')
                            substation = rng.choice(SUBSTATIONS) if V_kV >= 132 else 'N/A'

                            chunk = pd.DataFrame({
                                'voltage_kV':              np.full(n_dist, V_kV),
                                'current_A':               np.full(n_dist, round(I, 1)),
                                'distance_m':              distances,
                                'measurement_height_m':    np.full(n_dist, y_meas),
                                'height_m':                np.full(n_dist, h),
                                'phase_spacing_m':         np.full(n_dist, spacing),
                                'span_length_m':           np.full(n_dist, span),
                                'conductor_radius_cm':     np.full(n_dist, round(r_eq * 100, 3)),
                                'conductor_diameter_m':    np.full(n_dist, cond_d),
                                'bundle_count':            np.full(n_dist, n_sub),
                                'conductor_weight_kg_m':   np.full(n_dist, w_kg),
                                'sag_m':                   np.full(n_dist, round(sag, 3)),
                                'ground_clearance_m':      np.full(n_dist, round(gc, 2)),
                                'configuration':           np.full(n_dist, cfg_name),
                                'feeder':                  np.full(n_dist, feeder),
                                'substation':              np.full(n_dist, substation),
                                'temperature_C':           np.full(n_dist, Ta),
                                'conductor_temp_C':        np.full(n_dist, round(Tc, 1)),
                                'humidity_pct':            np.full(n_dist, hum),
                                'elevation_m':             np.full(n_dist, elev),
                                'wind_speed_ms':           np.full(n_dist, ws),
                                'soil_resistivity_ohm_m':  np.full(n_dist, sr),
                                'solar_irradiance_W_m2':   np.full(n_dist, sol),
                                'ice_thickness_mm':        np.full(n_dist, ice),
                                'weather':                 np.full(n_dist, weather),
                                'time_of_day':             np.full(n_dist, tod),
                                'season':                  np.full(n_dist, season),
                                'load_factor':             np.full(n_dist, round(lf, 3)),
                                'power_factor':            pf,
                                'active_power_MW':         P_MW,
                                'B_field_uT':              B_meas,
                                'E_field_V_m':             E_meas,
                                'H_field_A_m':             H_meas,
                                'B_field_clean_uT':        B_uT,
                                'E_field_clean_V_m':       E_Vm,
                                'corona_onset_kV_cm':      np.full(n_dist, round(Ec, 3)),
                                'surface_gradient_kV_cm':  np.full(n_dist, round(Es, 4)),
                                'corona_ratio':            np.full(n_dist, round(corona_ratio, 4)),
                                'frequency_Hz':            np.full(n_dist, FREQ),
                                'profile_type':            np.full(n_dist, 'Overhead Transmission'),
                                'circuit_type':            np.full(n_dist, 'Double-Circuit 3ph AC' if is_dc else '3-Phase AC'),
                                'phase_angle_deg':         np.full(n_dist, round(np.degrees(phi0), 2)),
                            })

                            chunks.append(chunk)
                            row_count += n_dist

                            if row_count % 200_000 == 0:
                                print(f"    ... {row_count:>12,} rows")

    print(f"\n  Concatenating {len(chunks):,} chunks ...")
    df = pd.concat(chunks, ignore_index=True)

    elapsed = time.time() - t0

    # Engineered features
    df['voltage_current_product'] = df['voltage_kV'] * df['current_A']
    df['height_to_spacing_ratio'] = df['height_m'] / df['phase_spacing_m']
    df['distance_to_height_ratio'] = df['distance_m'] / df['height_m']
    df['sag_to_span_ratio']        = df['sag_m'] / df['span_length_m']
    df['log_distance']             = np.log1p(df['distance_m'])
    df['sqrt_distance']            = np.sqrt(df['distance_m'])
    df['inv_distance']             = 1.0 / (df['distance_m'] + 1)
    df['inv_distance_sq']          = 1.0 / (df['distance_m']**2 + 1)
    df['temp_humidity_interaction']= df['temperature_C'] * df['humidity_pct']
    df['power_density']            = df['active_power_MW'] / (df['distance_m'] * df['height_m'] + 1)

    # ICNIRP compliance flag
    df['ICNIRP_E_exceeded'] = (df['E_field_V_m'] > ICNIRP_E_LIMIT).astype(int)
    df['ICNIRP_B_exceeded'] = (df['B_field_uT']  > ICNIRP_B_LIMIT).astype(int)

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 74)
    print(f"  Dataset generated in {elapsed:.1f}s")
    print(f"  Rows: {len(df):>12,}   |   Columns: {len(df.columns)}")
    print(f"  File: {output_file}")
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Memory: {mem:.1f} MB")
    print("=" * 74)

    print("\n  Field statistics (Narda EHP-50F readings):")
    for col in ['B_field_uT', 'E_field_V_m', 'H_field_A_m']:
        s = df[col]
        print(f"    {col:25s}  min={s.min():.4f}  median={s.median():.4f}"
              f"  mean={s.mean():.4f}  max={s.max():.4f}")

    print(f"\n  Voltage distribution:")
    print(df['voltage_kV'].value_counts().sort_index().to_string())

    print(f"\n  ICNIRP exceedances:  E={df['ICNIRP_E_exceeded'].sum():,}"
          f"   B={df['ICNIRP_B_exceeded'].sum():,}")

    return df


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
SAMPLE_CONFIG = textwrap.dedent("""\
{
  "_comment": "Custom line config example.  voltage_kV is the dict key.",
  "220": {
    "voltage_kV": 220,
    "I_base_A": 800,
    "height_m": 30,
    "phase_spacing_m": 7.0,
    "span_m": 350,
    "sub_radius_m": 0.015,
    "n_sub": 2,
    "bundle_spacing_m": 0.40,
    "w_kg_m": 1.6,
    "conductor_diam_m": 0.030,
    "R_per_km": 0.04,
    "rated_tensile_N": 120000,
    "configs": ["horizontal", "delta"],
    "feeders": ["Line-A", "Line-B"]
  }
}
""")

def load_config(path: str) -> Dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items() if not k.startswith('_')}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='EMF Dataset Generator v4 – Universal, Instrument-Aware',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Sample JSON config:\n{SAMPLE_CONFIG}")

    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with custom line parameters')
    parser.add_argument('--multiplier', type=int, default=5,
                        help='Phase-replica multiplier per scenario (default 5)')
    parser.add_argument('--output', type=str, default='grid_emf_dataset.csv',
                        help='Output CSV filename')
    parser.add_argument('--heights', type=float, nargs='+', default=[1.0, 1.5],
                        help='Measurement heights in metres (default: 1.0 1.5)')
    parser.add_argument('--dump-config', action='store_true',
                        help='Print sample JSON config and exit')

    args = parser.parse_args()

    if args.dump_config:
        print(SAMPLE_CONFIG)
        return

    cfg = load_config(args.config) if args.config else None

    df = generate_dataset(
        line_configs=cfg,
        multiplier=max(1, args.multiplier),
        output_file=args.output,
        measurement_heights=args.heights,
    )

    print("\nSample rows:")
    print(df.head(3).to_string())


if __name__ == '__main__':
    main()
