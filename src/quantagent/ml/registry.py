"""Versioned model registry on disk.

  <root>/index.json                {"current": version, "versions": [{version, created_at, kind, ...}]}
  <root>/<version>/model.joblib    the selected model (LogModel)
  <root>/<version>/har.joblib      HAR-RV refitted on the same rows (served as a baseline)
  <root>/<version>/config.json     features, hyperparameters, seed, data version, sha256 of each file
  <root>/<version>/metrics.json    walk-forward results the selection was based on

Files are loaded only if their sha256 matches config.json.
"""
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,  # noqa: S607
                              text=True, timeout=10, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a git checkout
        return None


def save(root: Path, version: str, objects: dict, config: dict, metrics: dict) -> Path:
    d = root / version
    d.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, obj in objects.items():
        joblib.dump(obj, d / f"{name}.joblib")
        hashes[f"{name}.joblib"] = _sha(d / f"{name}.joblib")
    config = {**config, "version": version, "files": hashes,
              "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"versions": []}
    index["versions"].append({k: config.get(k) for k in ("version", "created_at", "model_kind", "data_nature",
                                                         "data_version", "data_as_of")})
    index["current"] = version
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return d


def current_version(root: Path) -> str | None:
    p = root / "index.json"
    return json.loads(p.read_text(encoding="utf-8")).get("current") if p.exists() else None


def load(root: Path, version: str | None = None) -> tuple[dict, dict, dict]:
    """-> (objects by name, config, metrics). Raises FileNotFoundError / ValueError."""
    version = version or current_version(root)
    if not version:
        raise FileNotFoundError(f"no model registered under {root}")
    d = root / version
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    objects = {}
    for fname, sha in config["files"].items():
        if _sha(d / fname) != sha:
            raise ValueError(f"{d / fname} does not match its recorded sha256")
        objects[fname.removesuffix(".joblib")] = joblib.load(d / fname)  # noqa: S301 - hash-checked, own files
    return objects, config, metrics
