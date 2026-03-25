"""
Phase 1 — Digital Twin Simulation (simulation.py)

Generates 20,160 rows of synthetic sensor data across 10 global data centers
using Monte Carlo methods. Produces two DataFrames:

  - hydrological: coolant flow, pressure, temperature, water stress index,
                  anomaly flags (clog / leak / flood)
  - computational: CPU utilisation, power draw, PUE, WUE

Sampling interval : 10 minutes
Simulation window : 14 days  (14 × 24 × 6 = 2,016 rows per DC × 10 DCs = 20,160)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data-center catalogue (CBRE 2024 / README)
# ---------------------------------------------------------------------------

_DATA_CENTERS = [
    {"dc_id": "DC_N_Virginia", "capacity_mw": 3_500, "wue_baseline": 0.90, "stress": "HIGH"},
    {"dc_id": "DC_Beijing",    "capacity_mw": 1_799, "wue_baseline": 1.60, "stress": "HIGH"},
    {"dc_id": "DC_London",     "capacity_mw": 1_600, "wue_baseline": 0.45, "stress": "LOW"},
    {"dc_id": "DC_Singapore",  "capacity_mw": 1_400, "wue_baseline": 1.50, "stress": "HIGH"},
    {"dc_id": "DC_Tokyo",      "capacity_mw": 1_350, "wue_baseline": 0.55, "stress": "LOW"},
    {"dc_id": "DC_Frankfurt",  "capacity_mw": 1_200, "wue_baseline": 0.70, "stress": "MEDIUM"},
    {"dc_id": "DC_Shanghai",   "capacity_mw": 1_100, "wue_baseline": 1.40, "stress": "HIGH"},
    {"dc_id": "DC_Sydney",     "capacity_mw":   900, "wue_baseline": 1.20, "stress": "HIGH"},
    {"dc_id": "DC_Dallas",     "capacity_mw":   850, "wue_baseline": 1.80, "stress": "HIGH"},
    {"dc_id": "DC_Phoenix",    "capacity_mw":   800, "wue_baseline": 2.10, "stress": "EXTREME"},
]

# Anomaly injection probabilities per stress level (per 10-min tick)
_ANOMALY_PROB: Dict[str, Dict[str, float]] = {
    "LOW":     {"clog": 0.0005, "leak": 0.0003, "flood": 0.0001},
    "MEDIUM":  {"clog": 0.0015, "leak": 0.0010, "flood": 0.0005},
    "HIGH":    {"clog": 0.0040, "leak": 0.0030, "flood": 0.0015},
    "EXTREME": {"clog": 0.0080, "leak": 0.0060, "flood": 0.0035},
}

_ROWS_PER_DC = 2_016   # 14 days × 24 h × 6 ticks/h
_INTERVAL_MIN = 10


def _diurnal_load(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return a 0-1 diurnal workload envelope (peaks at ~14:00 local time)."""
    hour = timestamps.hour.to_numpy(float) + timestamps.minute.to_numpy(float) / 60.0
    # Two humps: business-hours peak + a smaller overnight trough
    envelope = (
        0.55
        + 0.30 * np.sin(np.pi * (hour - 6) / 12) ** 2   # daytime peak
        - 0.10 * np.sin(np.pi * (hour - 0) / 12) ** 2   # overnight dip
    )
    return np.clip(envelope, 0.2, 1.0)


def _simulate_dc(dc: dict, timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one data centre for the given timestamps.

    Returns
    -------
    hydro_df : pd.DataFrame
        Hydrological sensor readings.
    comp_df  : pd.DataFrame
        Computational metrics.
    """
    n = len(timestamps)
    dc_id = dc["dc_id"]
    stress = dc["stress"]
    capacity_mw = dc["capacity_mw"]
    wue_base = dc["wue_baseline"]
    anom_p = _ANOMALY_PROB[stress]

    # ------------------------------------------------------------------
    # Computational layer — CPU utilisation & power
    # ------------------------------------------------------------------
    base_load = _diurnal_load(timestamps)
    noise_load = rng.normal(0, 0.04, n)
    cpu_util = np.clip(base_load + noise_load, 0.05, 1.0)

    # Power draw follows utilisation with a floor at ~30 % (idle power)
    power_mw = capacity_mw * (0.30 + 0.70 * cpu_util) + rng.normal(0, capacity_mw * 0.005, n)
    power_mw = np.clip(power_mw, 0, capacity_mw)

    # PUE: add small Monte Carlo perturbation around a DC-specific mean
    pue_mean = 1.20 + (wue_base - 0.45) * 0.12   # higher WUE → slightly higher PUE
    pue = np.clip(rng.normal(pue_mean, 0.02, n), 1.05, 2.0)

    # ------------------------------------------------------------------
    # Hydrological layer — coolant circuit sensors
    # ------------------------------------------------------------------
    # Nominal flow scales with power draw (L/min = kW × WUE / 60)
    flow_nominal = (power_mw * 1_000) * wue_base / 60.0   # L/min
    flow_noise = rng.normal(0, flow_nominal * 0.02, n)
    flow_rate = flow_nominal + flow_noise   # L/min

    # Inlet / outlet temperatures
    inlet_temp = np.clip(rng.normal(18.0, 1.0, n), 12.0, 28.0)  # °C
    delta_t = wue_base * 3.5 + rng.normal(0, 0.5, n)             # °C rise
    outlet_temp = inlet_temp + delta_t

    # Pressure (bar) — nominal ~4 bar, slight variation
    pressure = np.clip(rng.normal(4.0, 0.15, n), 2.5, 6.0)

    # Water stress index (0-1) — driven by WUE and current utilisation
    stress_idx = np.clip(
        (wue_base / 2.10) * cpu_util + rng.normal(0, 0.02, n),
        0.0, 1.0,
    )

    # ------------------------------------------------------------------
    # Monte Carlo anomaly injection
    # ------------------------------------------------------------------
    anomaly = np.full(n, "none", dtype=object)
    clog_mask  = rng.random(n) < anom_p["clog"]
    leak_mask  = rng.random(n) < anom_p["leak"]
    flood_mask = rng.random(n) < anom_p["flood"]

    # Apply anomaly effects to sensor readings
    flow_rate[clog_mask]  *= rng.uniform(0.3, 0.7, clog_mask.sum())   # flow drops
    pressure[clog_mask]   += rng.uniform(0.5, 1.5, clog_mask.sum())   # back-pressure rises
    flow_rate[leak_mask]  *= rng.uniform(0.7, 0.95, leak_mask.sum())  # slight flow loss
    pressure[leak_mask]   -= rng.uniform(0.3, 0.8, leak_mask.sum())   # pressure drops
    flow_rate[flood_mask] *= rng.uniform(1.2, 1.8, flood_mask.sum())  # excess flow

    # Label anomalies (flood > clog > leak in priority)
    anomaly[clog_mask]  = "clog"
    anomaly[leak_mask]  = "leak"
    anomaly[flood_mask] = "flood"

    # WUE realised (affected by anomalies)
    wue_realised = np.where(
        anomaly != "none",
        wue_base * rng.uniform(1.05, 1.30, n),
        wue_base + rng.normal(0, wue_base * 0.03, n),
    )
    wue_realised = np.clip(wue_realised, 0.10, 5.0)

    # ------------------------------------------------------------------
    # Assemble DataFrames
    # ------------------------------------------------------------------
    hydro_df = pd.DataFrame({
        "timestamp":        timestamps,
        "dc_id":            dc_id,
        "stress_level":     stress,
        "flow_rate_lpm":    np.round(flow_rate, 2),
        "pressure_bar":     np.round(np.clip(pressure, 0, 8), 3),
        "inlet_temp_c":     np.round(inlet_temp, 2),
        "outlet_temp_c":    np.round(outlet_temp, 2),
        "water_stress_idx": np.round(stress_idx, 4),
        "anomaly":          anomaly,
        "wue_realised":     np.round(wue_realised, 4),
    })

    comp_df = pd.DataFrame({
        "timestamp":   timestamps,
        "dc_id":       dc_id,
        "cpu_util":    np.round(cpu_util, 4),
        "power_mw":    np.round(power_mw, 2),
        "capacity_mw": capacity_mw,
        "pue":         np.round(pue, 4),
        "wue":         np.round(wue_realised, 4),
    })

    return hydro_df, comp_df


def run_simulation(
    output_dir: str = "shcolite_data",
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> Dict[str, pd.DataFrame]:
    """Run the SHCO-Lite Digital Twin simulation.

    Generates ``_ROWS_PER_DC`` rows per data centre (10-minute intervals over
    14 days) for all 10 modelled data centres, producing 20,160 total rows.

    Parameters
    ----------
    output_dir : str
        Directory where CSV outputs are written.
    seed : int
        NumPy random seed for reproducibility.
    start_date : str
        ISO-8601 start date for the simulation window.

    Returns
    -------
    dict with keys:
        ``"hydrological"`` — pd.DataFrame of hydrological sensor readings
        ``"computational"`` — pd.DataFrame of computational metrics
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start=start_date,
        periods=_ROWS_PER_DC,
        freq=f"{_INTERVAL_MIN}min",
    )

    hydro_frames = []
    comp_frames  = []

    for dc in _DATA_CENTERS:
        h, c = _simulate_dc(dc, timestamps, rng)
        hydro_frames.append(h)
        comp_frames.append(c)

    hydro_df = pd.concat(hydro_frames, ignore_index=True)
    comp_df  = pd.concat(comp_frames,  ignore_index=True)

    # Sort by timestamp then dc_id for consistent ordering
    hydro_df = hydro_df.sort_values(["timestamp", "dc_id"]).reset_index(drop=True)
    comp_df  = comp_df.sort_values(["timestamp", "dc_id"]).reset_index(drop=True)

    # Persist to CSV
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hydro_df.to_csv(out_path / "hydrological.csv", index=False)
    comp_df.to_csv(out_path  / "computational.csv", index=False)

    print(
        f"Simulation complete — {len(hydro_df):,} rows written to '{output_dir}/'\n"
        f"  hydrological.csv : {len(hydro_df):,} rows × {hydro_df.shape[1]} cols\n"
        f"  computational.csv: {len(comp_df):,} rows × {comp_df.shape[1]} cols"
    )

    return {"hydrological": hydro_df, "computational": comp_df}
