"""Run output writing: results CSVs, traces, merged config, metadata."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from .config import config_hash


def _git_hash() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def write_run_outputs(out_dir: str | Path, cfg: Dict[str, Any], results: pd.DataFrame,
                      extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write grid results, the exact merged config, and run metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "grid_results.csv", index=False)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    import numpy, pandas  # versions for the record
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:                      # scipy is not required by the analysis
        scipy_version = None
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg),
        "git_hash": _git_hash(),
        "python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy_version,
    }
    meta.update(extra_meta or {})
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def write_trace(out_dir: str | Path, name: str, trace: Dict[str, np.ndarray],
                fmt: str = "parquet") -> Path:
    """Write a full per-iteration trace (every input + intermediate)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({k: v for k, v in trace.items() if isinstance(v, np.ndarray)})
    if fmt == "parquet":
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = out / f"{name}.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
    return path
