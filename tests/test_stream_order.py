"""The RNG consumption order must be enforced, not merely documented.

`sample_all` draws one array per configured distribution from a single seeded
Generator, so the ORDER of `config["distributions"]` decides which slice of
the stream each variable gets. Reorder that mapping and every sampled value
changes, silently, with no error and no diff in any individual parameter --
the model simply reports different numbers under the same seed.

`CANONICAL_DIST_ORDER` records the required order, and the module docstrings
say order matters, but before this file nothing checked it: `validate_config`
verified that each name was PRESENT, never that the order matched. A YAML
tidy-up, an alphabetical sort by a formatter, or an overlay introducing a new
distribution key would all have passed validation and changed the results.

These tests exist in two halves. The first half proves the order is
load-bearing (reordering really does move the numbers). The second proves the
guard catches it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aw_model.config import CANONICAL_DIST_ORDER, ConfigError, load_config, validate_config
from aw_model.distributions import sample_all, sample_dist

CONFIG_DIR = Path(__file__).parent.parent / "config"


def base_cfg() -> dict:
    return load_config(CONFIG_DIR / "base.yaml")


def test_the_shipped_config_is_in_canonical_order() -> None:
    assert tuple(base_cfg()["distributions"]) == CANONICAL_DIST_ORDER


def test_mapping_order_would_change_the_draws() -> None:
    """Establish the stakes: this is not a style rule.

    Drawn the way `sample_all` used to draw -- one array per distribution, in
    whatever order the mapping happens to be in -- a reordered block gives
    every variable a different slice of the same seeded stream. The assertion
    is on that raw behaviour, so it keeps proving the risk is real after
    `sample_all` itself was pinned to the canonical order.
    """
    dists = base_cfg()["distributions"]

    def draw_in(order: tuple[str, ...]) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(12345)
        return {name: sample_dist(rng, dists[name], 512, name) for name in order}

    straight = draw_in(CANONICAL_DIST_ORDER)
    swapped = draw_in(tuple(reversed(CANONICAL_DIST_ORDER)))

    differing = [
        name
        for name in CANONICAL_DIST_ORDER
        if not np.array_equal(straight[name], swapped[name])
    ]
    assert differing, "reordering the distribution block left every draw unchanged"


def test_a_reordered_distribution_block_is_rejected() -> None:
    """The guard itself: order is validated, not assumed."""
    cfg = base_cfg()
    dists = cfg["distributions"]
    cfg["distributions"] = {name: dists[name] for name in reversed(CANONICAL_DIST_ORDER)}
    with pytest.raises(ConfigError, match="order"):
        validate_config(cfg)


def test_two_swapped_neighbours_are_rejected() -> None:
    """A whole-block reversal is easy to spot; a formatter swapping two
    adjacent keys is the realistic accident."""
    cfg = base_cfg()
    dists = cfg["distributions"]
    order = list(CANONICAL_DIST_ORDER)
    order[3], order[4] = order[4], order[3]
    cfg["distributions"] = {name: dists[name] for name in order}
    with pytest.raises(ConfigError, match="order"):
        validate_config(cfg)


def test_an_extra_distribution_is_rejected() -> None:
    """Adding a distribution shifts every later variable's slice of the
    stream. That may be a legitimate model change, but it must be a
    deliberate one that also extends CANONICAL_DIST_ORDER."""
    cfg = base_cfg()
    cfg["distributions"]["some_new_input"] = {"dist": "fixed", "value": 1.0}
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_sample_all_consumes_the_canonical_order_regardless_of_mapping_order() -> None:
    """Defence in depth for callers that build a config without load_config."""
    cfg = base_cfg()
    dists = cfg["distributions"]
    reordered = {name: dists[name] for name in reversed(CANONICAL_DIST_ORDER)}

    straight = sample_all(np.random.default_rng(999), {"distributions": dists}, 256)
    from_reordered = sample_all(np.random.default_rng(999), {"distributions": reordered}, 256)

    for name in CANONICAL_DIST_ORDER:
        np.testing.assert_array_equal(
            straight[name],
            from_reordered[name],
            err_msg=f"{name}: sample_all is still following mapping order",
        )


@pytest.mark.parametrize(
    "overlay",
    [
        None,
        "scenarios/high_pr.yaml",
        "scenarios/fast_deploy.yaml",
        "scenarios/high_pr_fast_deploy.yaml",
        "correlation/plausible.yaml",
        "correlation/behavioural.yaml",
        "correlation/stress.yaml",
        "correlation/stress_fixed_pr.yaml",
        "severity/fatality_only.yaml",
        "severity/legacy_occupant.yaml",
        "severity/occupant_frontal.yaml",
    ],
)
def test_every_shipped_overlay_preserves_the_order(overlay: str | None) -> None:
    overlays = [] if overlay is None else [CONFIG_DIR / overlay]
    cfg = load_config(CONFIG_DIR / "base.yaml", overlays)
    assert tuple(cfg["distributions"]) == CANONICAL_DIST_ORDER
