"""Injury severity curves, parameterised so alternative curves can be selected
via config (harm-definition robustness, supervisor comment #41).

All curves are logistic in either impact speed (km/h) or delta-V. Named curves
carry their provenance; custom curves can be supplied from config as
{form: logistic, a: <intercept>, b: <slope>, speed_unit: kmh|mph}.

Named curve provenance (all verified against primary sources on 2026-07-04;
extraction notes in research/methods-citations-verified.md):

Worker (person on foot), impact speed:
- rosen_mais3plus: severe injury (MAIS3+), z = -4.6 + 0.078*v_kmh.
  Rosén, Källhammer, Eriksson, Nentwich, Fredriksson & Smith, "Pedestrian
  injury mitigation by autonomous braking" (ESV 09-0132; AAP 42(6) 2010,
  doi 10.1016/j.aap.2010.05.018), Table 1 (GIDAS, n=694, p<0.0001).
- rosen_fatality: fatality, z = -7.5 + 0.096*v_kmh. Same source/table
  (GIDAS, n=755 incl. 38 fatal).

Occupant (striking vehicle), delta-V:
- wang_mais3plus_all: MAIS3+, z = -6.9540 + 0.1637*dv_mph. Wang, J.-S.
  (2022), "MAIS(05/08) Injury Probability Curves as Functions of Delta V",
  NHTSA DOT HS 813 219, Table 4-1 (2010-2015 NASS-CDS, all crashes).
- wang_mais3plus_frontal: MAIS3+, z = -6.9774 + 0.1620*dv_mph. Same report,
  Table 4-3 (frontal crashes).
- wang_fatality_all: fatality, z = -8.9819 + 0.1603*dv_mph. Same report,
  Table 4-1.
- kahane_mais3plus (LEGACY): z = 0.1292*dv_mph - 5.5337. Used by v0.1; its
  claimed source (DOT HS 811 664/665 App. D) could NOT be verified, so it is
  retained ONLY for the v0.1 reproduction gate and must not be cited.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

MPH_PER_KMH = 1.0 / 1.609344

# Named curve registry: name -> (a, b, speed_unit)
# p = 1 / (1 + exp(-(a + b * v)))  with v in the stated unit
WORKER_CURVES: Dict[str, Dict[str, Any]] = {
    "rosen_mais3plus": {"a": -4.6, "b": 0.078, "speed_unit": "kmh"},
    "rosen_fatality": {"a": -7.5, "b": 0.096, "speed_unit": "kmh"},
}
OCCUPANT_CURVES: Dict[str, Dict[str, Any]] = {
    "wang_mais3plus_all": {"a": -6.9540, "b": 0.1637, "speed_unit": "mph"},
    "wang_mais3plus_frontal": {"a": -6.9774, "b": 0.1620, "speed_unit": "mph"},
    "wang_fatality_all": {"a": -8.9819, "b": 0.1603, "speed_unit": "mph"},
    "kahane_mais3plus": {"a": -5.5337, "b": 0.1292, "speed_unit": "mph"},  # legacy, reproduction only
}


def _logistic(v: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a + b * v
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 0.0, 1.0)


def _resolve(spec: Any, registry: Dict[str, Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """Resolve a curve spec (a registry name or a parameter mapping)."""
    if isinstance(spec, str):
        if spec not in registry:
            raise ValueError(f"Unknown {kind} curve '{spec}'. Known: {sorted(registry)}")
        return registry[spec]
    if isinstance(spec, dict):
        for key in ("a", "b"):
            if key not in spec:
                raise ValueError(f"Custom {kind} curve missing '{key}': {spec}")
        return {"a": float(spec["a"]), "b": float(spec["b"]),
                "speed_unit": str(spec.get("speed_unit", "kmh")).lower()}
    raise ValueError(f"Invalid {kind} curve spec: {spec!r}")


def p_worker(v_kmh: np.ndarray, curve: Any = "rosen_mais3plus") -> np.ndarray:
    """Probability of the harm outcome for a person on foot struck at v_kmh."""
    c = _resolve(curve, WORKER_CURVES, "worker")
    v = np.asarray(v_kmh, dtype=float)
    if c["speed_unit"] == "mph":
        v = v * MPH_PER_KMH
    return _logistic(v, c["a"], c["b"])


def p_occupant(delta_v_kmh: np.ndarray, curve: Any = "wang_mais3plus_all") -> np.ndarray:
    """Probability of the harm outcome for an occupant given delta-V (km/h input)."""
    c = _resolve(curve, OCCUPANT_CURVES, "occupant")
    dv = np.asarray(delta_v_kmh, dtype=float)
    if c["speed_unit"] == "mph":
        dv = dv * MPH_PER_KMH
    return _logistic(dv, c["a"], c["b"])


# Backwards-compatible aliases (v0.1 names)
def p_worker_ais3plus(v_kmh: np.ndarray) -> np.ndarray:
    return p_worker(v_kmh, "rosen_mais3plus")


def p_occ_ais3plus(delta_v_kmh: np.ndarray) -> np.ndarray:
    return p_occupant(delta_v_kmh, "kahane_mais3plus")
