"""Configuration loading with base + overlay deep-merge.

The base config carries every default; scenario overlays are minimal diffs.
Every run stamps the fully-merged config into its output directory so any
result can be traced to the exact inputs that produced it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

REQUIRED_TOP_KEYS = ("meta", "grid", "monte_carlo", "model", "distributions")

# Distribution names sampled by the model, in the canonical (legacy) order.
# Order matters: reproducing the v0.1 results bit-for-bit requires consuming
# the RNG stream in exactly this order.
CANONICAL_DIST_ORDER = (
    "d_aw_m",
    "v_walk_m_s",
    "t_handle_s",
    "mu_v_kmh",
    "sigma_v_kmh",
    "v_cap_kmh",
    "p_R",
    "deltaV_kmh",
    "a_comfort_m_s2",
    "r_E_per_veh_km",
    "k_rE",
    "alpha_lat_m_inv",
    "c_work_m",
    "c_deploy_m",
    "k_E",
    "L_worker_m",
    "L_lcv_m",
    "L_worker_deploy_m",
    "m_striking_kg",
    "m_lcv_kg",
)


class ConfigError(ValueError):
    pass


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins on leaves)."""
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            # A distribution spec that changes family must REPLACE, not merge —
            # otherwise stale parameters (e.g. min/max) survive under a 'fixed'.
            if "dist" in val:
                out[key] = copy.deepcopy(val)
            else:
                out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(base_path: str | Path, overlays: Optional[Iterable[str | Path]] = None) -> Dict[str, Any]:
    """Load base YAML config and deep-merge any overlay files on top, in order."""
    base_path = Path(base_path)
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ConfigError(f"Base config {base_path} did not parse to a mapping")
    for ov_path in overlays or []:
        ov = yaml.safe_load(Path(ov_path).read_text(encoding="utf-8"))
        if not isinstance(ov, dict):
            raise ConfigError(f"Overlay {ov_path} did not parse to a mapping")
        cfg = _deep_merge(cfg, ov)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"Config missing required sections: {missing}")
    dists = cfg["distributions"]
    missing_d = [name for name in CANONICAL_DIST_ORDER if name not in dists]
    if missing_d:
        raise ConfigError(f"Config missing distributions: {missing_d}")
    for name, spec in dists.items():
        if "dist" not in spec:
            raise ConfigError(f"Distribution '{name}' missing 'dist' field")
    corr = cfg.get("correlation", {}) or {}
    for pair in corr.get("pairs", []) or []:
        if len(pair) != 3:
            raise ConfigError(f"Correlation pair must be [var1, var2, rho]: {pair}")
        v1, v2, rho = pair
        for v in (v1, v2):
            if v not in dists:
                raise ConfigError(f"Correlation names unknown distribution '{v}'")
        if not (-1.0 < float(rho) < 1.0):
            raise ConfigError(f"Correlation rho out of range (-1, 1): {pair}")


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of the fully-merged config (for output metadata)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
