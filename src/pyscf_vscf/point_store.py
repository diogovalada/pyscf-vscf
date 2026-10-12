"""Persistent, resumable storage of electronic point results (SQLite).

One store holds results for any number of providers and plans. A result is
keyed by its point causal fingerprint, which already includes the provider's
scientific fingerprint, so results from different methods never collide.
Only the coordinating process writes: worker processes return outcomes and
the coordinator records each one in a single transaction together with its
attempt record. A successful result is never overwritten; a later result with
the same key but a different scientific fingerprint is recorded as a conflict.
Failed and interrupted attempts are kept as history and never occupy the
result slot.

The store supports one coordinating process at a time on local storage.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._identity import canonical_json, payload_fingerprint
from .artifacts.electronic import request_to_json, result_from_json, result_to_json
from .contracts import ElectronicPointRequest, ElectronicResult

STORE_SCHEMA = "pyscf-vscf-point-store"
STORE_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS providers (
    fingerprint TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS points (
    point_key TEXT PRIMARY KEY,
    provider_fingerprint TEXT NOT NULL REFERENCES providers(fingerprint),
    request TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    point_key TEXT PRIMARY KEY REFERENCES points(point_key),
    scientific_fingerprint TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    plan_fingerprint TEXT,
    provider_fingerprint TEXT NOT NULL,
    n_points INTEGER NOT NULL,
    policy TEXT NOT NULL,
    status TEXT NOT NULL,
    started REAL NOT NULL,
    finished REAL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    point_key TEXT NOT NULL,
    status TEXT NOT NULL,
    started REAL NOT NULL,
    finished REAL,
    error_type TEXT,
    message TEXT,
    traceback TEXT
);
CREATE INDEX IF NOT EXISTS attempts_point ON attempts(point_key, attempt_id);
"""


class StoreConflictError(RuntimeError):
    """A stored point has a different scientific result than a new evaluation."""


@dataclass(frozen=True)
class AttemptRecord:
    """One recorded evaluation attempt of a point."""

    attempt_id: int
    run_id: str
    point_key: str
    status: str
    error_type: str | None
    message: str | None
    traceback: str | None


class PointStore:
    """SQLite store of electronic point results and evaluation attempts.

    Open with a path (created if missing). Use as a context manager or call
    ``close``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), isolation_level=None)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(_SCHEMA_SQL)
        self._check_schema()

    def close(self) -> None:
        """Close the database connection."""

        self._connection.close()

    def __enter__(self) -> PointStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def register_provider(self, fingerprint: str, payload: Mapping[str, Any]) -> None:
        """Record a provider's scientific payload; a stored payload must match."""

        text = canonical_json(dict(payload))
        if payload_fingerprint(dict(payload)) != fingerprint:
            raise ValueError("Provider payload does not match its fingerprint")
        row = self._connection.execute(
            "SELECT payload FROM providers WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO providers (fingerprint, payload) VALUES (?, ?)", (fingerprint, text)
            )
        elif row[0] != text:
            raise StoreConflictError("Stored provider payload differs for the same fingerprint")

    def provider_payload(self, fingerprint: str) -> Mapping[str, Any] | None:
        """Return the stored scientific payload of a provider, if any."""

        row = self._connection.execute(
            "SELECT payload FROM providers WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def begin_run(
        self,
        *,
        provider_fingerprint: str,
        n_points: int,
        plan_fingerprint: str | None = None,
        policy: Mapping[str, Any] | None = None,
    ) -> str:
        """Start a run; attempts left in progress by earlier runs become interrupted."""

        now = time.time()
        with self._transaction():
            self._connection.execute(
                "UPDATE attempts SET status = 'interrupted', finished = ? "
                "WHERE status = 'in_progress'",
                (now,),
            )
            self._connection.execute(
                "UPDATE runs SET status = 'interrupted', finished = ? WHERE status = 'running'",
                (now,),
            )
            run_id = uuid.uuid4().hex
            self._connection.execute(
                "INSERT INTO runs (run_id, plan_fingerprint, provider_fingerprint, n_points, "
                "policy, status, started) VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (
                    run_id,
                    plan_fingerprint,
                    provider_fingerprint,
                    int(n_points),
                    canonical_json(dict(policy or {})),
                    now,
                ),
            )
        return run_id

    def finish_run(self, run_id: str, status: str) -> None:
        """Mark a run finished with ``status`` (complete, incomplete, interrupted)."""

        with self._transaction():
            self._connection.execute(
                "UPDATE attempts SET status = 'interrupted', finished = ? "
                "WHERE run_id = ? AND status = 'in_progress'",
                (time.time(), run_id),
            )
            self._connection.execute(
                "UPDATE runs SET status = ?, finished = ? WHERE run_id = ?",
                (status, time.time(), run_id),
            )

    def begin_attempt(self, run_id: str, point_key: str) -> int:
        """Record that a point evaluation started; return the attempt id."""

        cursor = self._connection.execute(
            "INSERT INTO attempts (run_id, point_key, status, started) "
            "VALUES (?, ?, 'in_progress', ?)",
            (run_id, point_key, time.time()),
        )
        attempt_id = cursor.lastrowid
        assert attempt_id is not None
        return int(attempt_id)

    def record_success(
        self,
        attempt_id: int,
        request: ElectronicPointRequest,
        result: ElectronicResult,
    ) -> None:
        """Store a validated result and close its attempt in one transaction.

        Raises ``StoreConflictError`` (after recording the conflict) if a
        different scientific result is already stored under the same key.
        """

        key = result.point_causal_fingerprint
        now = time.time()
        with self._transaction():
            self._connection.execute(
                "INSERT OR IGNORE INTO points (point_key, provider_fingerprint, request) "
                "VALUES (?, ?, ?)",
                (
                    key,
                    result.provider_scientific_fingerprint,
                    canonical_json(request_to_json(request)),
                ),
            )
            existing = self._connection.execute(
                "SELECT scientific_fingerprint FROM results WHERE point_key = ?", (key,)
            ).fetchone()
            scientific = result.scientific_fingerprint()
            if existing is not None and existing[0] != scientific:
                self._connection.execute(
                    "UPDATE attempts SET status = 'conflict', finished = ?, message = ? "
                    "WHERE attempt_id = ?",
                    (now, "scientific result differs from the stored result", attempt_id),
                )
                conflict = True
            else:
                conflict = False
                if existing is None:
                    self._connection.execute(
                        "INSERT INTO results (point_key, scientific_fingerprint, "
                        "content_fingerprint, payload, created) VALUES (?, ?, ?, ?, ?)",
                        (
                            key,
                            scientific,
                            result.content_fingerprint(),
                            json.dumps(result_to_json(result), sort_keys=True),
                            now,
                        ),
                    )
                self._connection.execute(
                    "UPDATE attempts SET status = 'succeeded', finished = ? WHERE attempt_id = ?",
                    (now, attempt_id),
                )
        if conflict:
            raise StoreConflictError(
                f"Point {key[:12]}... already has a different stored scientific result"
            )

    def record_failure(
        self,
        attempt_id: int,
        *,
        error_type: str,
        message: str,
        traceback: str = "",
        status: str = "failed",
    ) -> None:
        """Close an attempt as failed (or interrupted) with its error details."""

        self._connection.execute(
            "UPDATE attempts SET status = ?, finished = ?, error_type = ?, message = ?, "
            "traceback = ? WHERE attempt_id = ?",
            (status, time.time(), error_type, message, traceback, attempt_id),
        )

    def get_result(self, point_key: str) -> ElectronicResult | None:
        """Return the verified stored result for a point key, if any."""

        row = self._connection.execute(
            "SELECT payload, content_fingerprint FROM results WHERE point_key = ?", (point_key,)
        ).fetchone()
        if row is None:
            return None
        result = result_from_json(json.loads(row[0]))
        if result.content_fingerprint() != row[1] or result.point_causal_fingerprint != point_key:
            raise ValueError(f"Stored result for point {point_key[:12]}... is corrupt")
        return result

    def latest_attempt(self, point_key: str) -> AttemptRecord | None:
        """Return the most recent attempt for a point key, if any."""

        row = self._connection.execute(
            "SELECT attempt_id, run_id, point_key, status, error_type, message, traceback "
            "FROM attempts WHERE point_key = ? ORDER BY attempt_id DESC LIMIT 1",
            (point_key,),
        ).fetchone()
        return None if row is None else AttemptRecord(*row)

    def attempts(self, point_key: str) -> tuple[AttemptRecord, ...]:
        """Return every attempt of a point key, oldest first."""

        rows = self._connection.execute(
            "SELECT attempt_id, run_id, point_key, status, error_type, message, traceback "
            "FROM attempts WHERE point_key = ? ORDER BY attempt_id",
            (point_key,),
        ).fetchall()
        return tuple(AttemptRecord(*row) for row in rows)

    def stored_keys(self, keys: Iterable[str]) -> frozenset[str]:
        """Return the subset of ``keys`` that have stored results."""

        found = set()
        for key in keys:
            row = self._connection.execute(
                "SELECT 1 FROM results WHERE point_key = ?", (key,)
            ).fetchone()
            if row is not None:
                found.add(key)
        return frozenset(found)

    def _check_schema(self) -> None:
        rows = dict(self._connection.execute("SELECT key, value FROM meta").fetchall())
        if not rows:
            with self._transaction():
                self._connection.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema', ?), ('schema_version', ?)",
                    (STORE_SCHEMA, str(STORE_SCHEMA_VERSION)),
                )
            return
        if rows.get("schema") != STORE_SCHEMA or rows.get("schema_version") != str(
            STORE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"{self.path} is not a pyscf-vscf point store of schema version "
                f"{STORE_SCHEMA_VERSION}"
            )

    def _transaction(self) -> _Transaction:
        return _Transaction(self._connection)


class _Transaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self._connection.execute("COMMIT")
        else:
            self._connection.execute("ROLLBACK")


__all__ = ["AttemptRecord", "PointStore", "StoreConflictError"]
