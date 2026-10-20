"""Read-only database engines.

SQLite: opened with mode=ro, PRAGMA query_only=ON, and a progress handler that aborts
statements running longer than the configured timeout.
Postgres: read-only transactions and a statement_timeout (connect with a SELECT-only role).
"""
import time
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url


def sqlite_path(db_url: str) -> Path:
    url = make_url(db_url)
    if not url.drivername.startswith("sqlite") or not url.database:
        raise ValueError(f"not a file-backed sqlite URL: {url.drivername}")
    return Path(url.database).resolve()


def make_readonly_engine(db_url: str, timeout_s: float = 10.0) -> Engine:
    url = make_url(db_url)
    if url.drivername.startswith("sqlite"):
        path = sqlite_path(db_url)
        if not path.exists():
            raise FileNotFoundError(f"database not found: {path} (run scripts/ingest.py)")
        engine = create_engine(
            f"sqlite+pysqlite:///file:{path.as_posix()}?mode=ro&uri=true",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, record):
            dbapi_conn.execute("PRAGMA query_only=ON")

            def _abort_if_slow() -> int:
                t0 = record.info.get("t0")
                return 1 if t0 is not None and time.monotonic() - t0 > timeout_s else 0

            dbapi_conn.set_progress_handler(_abort_if_slow, 10_000)

        @event.listens_for(engine, "before_cursor_execute")
        def _start_clock(conn, cursor, statement, params, context, executemany):
            conn.info["t0"] = time.monotonic()

        return engine

    if url.drivername.startswith("postgresql"):
        ms = int(timeout_s * 1000)
        return create_engine(
            url,
            connect_args={"options": f"-c default_transaction_read_only=on -c statement_timeout={ms}"},
            pool_pre_ping=True,
        )
    raise ValueError(f"unsupported database: {url.drivername}")
