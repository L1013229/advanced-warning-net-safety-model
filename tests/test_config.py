"""Config loading and overlay-merge tests."""
from pathlib import Path

import pytest

from aw_model.config import ConfigError, config_hash, load_config

CONFIG_DIR = Path(__file__).parent.parent / "config"


def test_base_loads_and_validates():
    cfg = load_config(CONFIG_DIR / "base.yaml")
    assert cfg["meta"]["tag"] == "baseline"
    assert cfg["distributions"]["p_R"]["max"] == 0.30


def test_overlay_merges_only_diffs():
    cfg = load_config(CONFIG_DIR / "base.yaml", [CONFIG_DIR / "scenarios/high_pr.yaml"])
    assert cfg["meta"]["tag"] == "highPR"
    assert cfg["distributions"]["p_R"] == {"dist": "fixed", "value": 0.6}
    # Untouched keys survive from base
    assert cfg["distributions"]["d_aw_m"]["mode"] == 50
    assert cfg["monte_carlo"]["seed"] == 12345


def test_overlay_stacking_order():
    cfg = load_config(CONFIG_DIR / "base.yaml",
                      [CONFIG_DIR / "scenarios/high_pr.yaml",
                       CONFIG_DIR / "correlation/plausible.yaml"])
    assert cfg["distributions"]["p_R"]["value"] == 0.6
    assert cfg["correlation"]["pairs"] == [["mu_v_kmh", "r_E_per_veh_km", 0.3]]
    assert cfg["meta"]["tag"] == "corr_plausible"  # last overlay wins the tag


def test_hash_stable_and_sensitive(tmp_path):
    cfg1 = load_config(CONFIG_DIR / "base.yaml")
    cfg2 = load_config(CONFIG_DIR / "base.yaml")
    assert config_hash(cfg1) == config_hash(cfg2)
    cfg2["monte_carlo"]["seed"] = 99
    assert config_hash(cfg1) != config_hash(cfg2)


def test_bad_correlation_pair_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("correlation:\n  pairs:\n    - [mu_v_kmh, not_a_var, 0.3]\n")
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / "base.yaml", [bad])
