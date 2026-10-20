"""Builds the synthetic fixture database once per session and points the tools at it."""
from pathlib import Path

import pytest

from quantagent.config import Settings
from quantagent.data.engine import make_readonly_engine
from quantagent.data.ingest import ingest
from quantagent.data.repository import Repository
from quantagent.tools import _base
from tests.fixtures import make_fixture as fx

RF = 0.05


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("qa")
    raw, ref = root / "raw", root / "reference"
    fx.build(raw, ref)
    db = root / "quantagent.db"
    settings = Settings(db_url=f"sqlite:///{db.as_posix()}", risk_free_annual=RF,
                        risk_free_source="test default", model_dir=str(root / "no-model"),
                        _env_file=None)
    summary = ingest(raw, db, ref, root / "rejected_rows.csv", settings, data_nature="synthetic")
    return {"root": root, "raw": raw, "ref": ref, "db": db, "settings": settings, "summary": summary,
            "rejected": root / "rejected_rows.csv"}


@pytest.fixture(scope="session")
def repo(built) -> Repository:
    return Repository(make_readonly_engine(built["settings"].db_url, built["settings"].timeout_s))


@pytest.fixture(autouse=True)
def services(built, repo):
    _base.set_services(_base.Services(built["settings"], repo))
    yield _base.get_services()
    _base.set_services(None)


def payload(exc) -> dict:
    """Decode a QuantError's JSON message."""
    import json
    return json.loads(str(exc))


@pytest.fixture
def fixture_module():
    return fx


ROOT = Path(__file__).resolve().parents[1]
