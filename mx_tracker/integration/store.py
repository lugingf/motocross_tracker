"""What the integration keeps on disk: jobs and the events they produced.

An event is written here before anything tries to send it, and a crash or a
dead connection loses nothing. Events are kept after they are acknowledged:
they are what laps are calculated from again after a correction.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    session_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    state TEXT NOT NULL,
    configuration TEXT NOT NULL DEFAULT '{}',
    race_start_ms INTEGER,
    corrections TEXT NOT NULL DEFAULT '[]',
    corrections_up_to INTEGER NOT NULL DEFAULT 0,
    next_seq INTEGER NOT NULL DEFAULT 1,
    acked_seq INTEGER NOT NULL DEFAULT 0,
    segment_id INTEGER NOT NULL DEFAULT 0,
    last_media_ms INTEGER NOT NULL DEFAULT 0,
    last_wall REAL NOT NULL DEFAULT 0,
    revision_counter INTEGER NOT NULL DEFAULT 0,
    finish_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS submissions (
    session_id TEXT NOT NULL,
    submission_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session_id, submission_id)
);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    media_ms INTEGER NOT NULL,
    segment_id INTEGER NOT NULL,
    participant_id TEXT NOT NULL DEFAULT '',
    plate_text TEXT NOT NULL DEFAULT '',
    plate_conf REAL,
    identity_source TEXT NOT NULL DEFAULT '',
    bbox TEXT,
    evidence_ref TEXT NOT NULL DEFAULT '',
    extra TEXT,
    PRIMARY KEY (session_id, seq)
);
"""

ACTIVE_STATES = ("queued", "running", "finishing")


@dataclass(slots=True)
class Job:
    session_id: str
    run_id: str
    state: str
    configuration: dict[str, Any]
    race_start_ms: int | None
    corrections: list[dict[str, Any]]
    corrections_up_to: int
    next_seq: int
    acked_seq: int
    segment_id: int
    last_media_ms: int
    last_wall: float
    revision_counter: int
    finish_requested: bool
    error: str


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript(SCHEMA)
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(events)")}
            if "evidence_done" not in columns:
                self._db.execute("ALTER TABLE events ADD COLUMN evidence_done INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            session_id=row["session_id"],
            run_id=row["run_id"],
            state=row["state"],
            configuration=json.loads(row["configuration"]),
            race_start_ms=row["race_start_ms"],
            corrections=json.loads(row["corrections"]),
            corrections_up_to=row["corrections_up_to"],
            next_seq=row["next_seq"],
            acked_seq=row["acked_seq"],
            segment_id=row["segment_id"],
            last_media_ms=row["last_media_ms"],
            last_wall=row["last_wall"],
            revision_counter=row["revision_counter"],
            finish_requested=bool(row["finish_requested"]),
            error=row["error"],
        )

    def get_job(self, session_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return self._job(row) if row else None

    def create_job(
        self, session_id: str, run_id: str, configuration: dict[str, Any], race_start_ms: int | None, state: str = "queued"
    ) -> Job:
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (session_id, run_id, state, configuration, race_start_ms, last_wall, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, run_id, state, json.dumps(configuration), race_start_ms, time.time(), time.time()),
            )
        return self.get_job(session_id)  # type: ignore[return-value]

    def list_jobs(self, states: tuple[str, ...] | None = None) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs").fetchall()
        jobs = [self._job(row) for row in rows]
        return [job for job in jobs if states is None or job.state in states]

    def set_state(self, session_id: str, state: str, error: str = "") -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET state = ?, error = ?, updated_at = ? WHERE session_id = ?",
                (state, error, time.time(), session_id),
            )

    def set_run(self, session_id: str, run_id: str, state: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET run_id = ?, state = ?, error = '', updated_at = ? WHERE session_id = ?",
                (run_id, state, time.time(), session_id),
            )

    def set_configuration(self, session_id: str, configuration: dict[str, Any], race_start_ms: int | None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET configuration = ?, race_start_ms = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(configuration), race_start_ms, time.time(), session_id),
            )

    def set_corrections(self, session_id: str, corrections: list[dict[str, Any]], up_to: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET corrections = ?, corrections_up_to = ? WHERE session_id = ?",
                (json.dumps(corrections), up_to, session_id),
            )

    def request_finish(self, session_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET finish_requested = 1 WHERE session_id = ?", (session_id,))

    def touch(self, session_id: str, media_ms: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET last_media_ms = MAX(last_media_ms, ?), last_wall = ? WHERE session_id = ?",
                (media_ms, time.time(), session_id),
            )

    def next_segment(self, session_id: str) -> int:
        with self._lock:
            self._db.execute("UPDATE jobs SET segment_id = segment_id + 1 WHERE session_id = ?", (session_id,))
            row = self._db.execute("SELECT segment_id FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return int(row["segment_id"])

    def next_revision(self, session_id: str) -> int:
        with self._lock:
            self._db.execute("UPDATE jobs SET revision_counter = revision_counter + 1 WHERE session_id = ?", (session_id,))
            row = self._db.execute("SELECT revision_counter FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return int(row["revision_counter"])

    def save_submission(self, session_id: str, submission: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO submissions (session_id, submission_id, status, payload) VALUES (?, ?, ?, ?)",
                (session_id, submission["submission_id"], submission["status"], json.dumps(submission)),
            )

    def pending_submissions(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM submissions WHERE session_id = ? ORDER BY rowid", (session_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def drop_submission(self, session_id: str, submission_id: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM submissions WHERE session_id = ? AND submission_id = ?", (session_id, submission_id)
            )

    def append_event(
        self,
        session_id: str,
        run_id: str,
        kind: str,
        media_ms: int,
        segment_id: int,
        participant_id: str = "",
        plate_text: str = "",
        plate_conf: float | None = None,
        identity_source: str = "",
        bbox: dict[str, float] | None = None,
        evidence_ref: str = "",
        extra: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT next_seq FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
                seq = int(row["next_seq"])
                self._db.execute(
                    "INSERT INTO events (session_id, seq, event_id, run_id, kind, media_ms, segment_id, participant_id,"
                    " plate_text, plate_conf, identity_source, bbox, evidence_ref, extra)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        seq,
                        f"{run_id[:8]}-{seq}",
                        run_id,
                        kind,
                        media_ms,
                        segment_id,
                        participant_id,
                        plate_text,
                        plate_conf,
                        identity_source,
                        json.dumps(bbox) if bbox is not None else None,
                        evidence_ref,
                        json.dumps(extra) if extra is not None else None,
                    ),
                )
                self._db.execute(
                    "UPDATE jobs SET next_seq = ?, last_media_ms = MAX(last_media_ms, ?), last_wall = ? WHERE session_id = ?",
                    (seq + 1, media_ms, time.time(), session_id),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return seq

    @staticmethod
    def _event(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "seq": row["seq"],
            "run_id": row["run_id"],
            "kind": row["kind"],
            "media_time_ms": row["media_ms"],
            "segment_id": row["segment_id"],
            "participant_id": row["participant_id"],
            "plate_text": row["plate_text"],
            "plate_confidence": row["plate_conf"],
            "identity_source": row["identity_source"],
            "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
            "evidence_ref": row["evidence_ref"],
            "extra": json.loads(row["extra"]) if row["extra"] else None,
        }

    def events_after(self, session_id: str, seq: int, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events WHERE session_id = ? AND seq > ? ORDER BY seq LIMIT ?", (session_id, seq, limit)
            ).fetchall()
        return [self._event(row) for row in rows]

    def events_upto(self, session_id: str, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events WHERE session_id = ? AND seq <= ? ORDER BY seq", (session_id, seq)
            ).fetchall()
        return [self._event(row) for row in rows]

    def events_awaiting_evidence(self, session_id: str, up_to_seq: int, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events WHERE session_id = ? AND seq <= ? AND evidence_done = 0 AND evidence_ref != ''"
                " AND kind = 'crossing' ORDER BY seq LIMIT ?",
                (session_id, up_to_seq, limit),
            ).fetchall()
        return [self._event(row) for row in rows]

    def mark_evidence_done(self, session_id: str, seqs: list[int]) -> None:
        with self._lock:
            self._db.executemany(
                "UPDATE events SET evidence_done = 1 WHERE session_id = ? AND seq = ?", [(session_id, seq) for seq in seqs]
            )

    def last_seq(self, session_id: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT next_seq FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return int(row["next_seq"]) - 1 if row else 0

    def pending_count(self, session_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM events WHERE session_id = ? AND seq > (SELECT acked_seq FROM jobs WHERE session_id = ?)",
                (session_id, session_id),
            ).fetchone()
        return int(row["n"])

    def acknowledge(self, session_id: str, seq: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET acked_seq = MAX(acked_seq, ?) WHERE session_id = ?", (seq, session_id)
            )
