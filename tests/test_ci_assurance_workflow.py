"""Shape of the CI assurance estate for this repository (daqt-server #5923).

One hosted job on a pinned image is the required check. It rests on the
properties pinned here: a read-only token, a bound, actions pinned to
commits, a checkout that persists no credential, a main run that is never
cancelled, and a register that accounts for every workflow file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
REGISTER = REPO_ROOT / "config" / "ci-assurance-register.json"
AGGREGATE = ".github/workflows/reproducibility.yml"
SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}(\s+#.*)?$")


def _workflow() -> dict:
    return yaml.safe_load((REPO_ROOT / AGGREGATE).read_text(encoding="utf-8"))


def _register() -> dict:
    return json.loads(REGISTER.read_text(encoding="utf-8"))


def test_the_register_accounts_for_every_workflow_file():
    tracked = sorted(str(p.relative_to(REPO_ROOT)) for p in WORKFLOWS.glob("*.yml"))
    register = _register()
    assert set(register["workflows"]) == set(tracked) == {AGGREGATE}
    for relative, row in register["workflows"].items():
        for field in ("origin", "protects", "consumer", "falsifier", "rollback"):
            assert row.get(field, "").strip(), (relative, field)
    assert register["required_check"] == {"context": "reproduce", "app_id": 15368, "aggregate": AGGREGATE}
    assert set(_workflow()["jobs"]) == {"reproduce"}


def test_the_job_is_hosted_on_a_pinned_image_bounded_and_read_only():
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["reproduce"]
    assert job["runs-on"] == "ubuntu-24.04", "a moving runner image is a moving BLAS"
    assert isinstance(job.get("timeout-minutes"), int)
    assert not (job.get("permissions") or {})


def test_every_action_is_pinned_and_the_checkout_persists_nothing():
    for step in _workflow()["jobs"]["reproduce"]["steps"]:
        uses = step.get("uses")
        if not uses:
            continue
        assert SHA_PIN.match(uses), uses
        if uses.startswith("actions/checkout@"):
            assert (step.get("with") or {}).get("persist-credentials") is False


def test_a_main_run_is_never_cancelled():
    workflow = _workflow()
    on = workflow.get(True) or workflow.get("on")
    assert on["push"] == {"branches": ["main"]}
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    assert "github.sha" in concurrency["group"]


def test_the_determinism_knobs_and_the_byte_identical_check_are_still_there():
    workflow = _workflow()
    for knob in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert knob in workflow["env"], knob
    runs = [step.get("run", "") for step in workflow["jobs"]["reproduce"]["steps"]]
    assert any("make_results_of_record.py --check" in run for run in runs)
    assert any("--no-deps -r requirements-lock.txt" in run for run in runs)


def test_dependabot_keeps_the_pins_moving():
    config = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    ecosystems = {update["package-ecosystem"] for update in config["updates"]}
    assert ecosystems == {"github-actions", "pip"}
    assert _register()["dependabot"]["ecosystems"] == sorted(ecosystems)
