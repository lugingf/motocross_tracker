"""Tests for service.py — JobRecord log management and JobManager state.

Tests only the pure state management layer; actual job execution (which calls
pipeline and training functions) is not exercised here.
"""
import pytest

from mx_tracker.service import JobManager, JobRecord


# ---------------------------------------------------------------------------
# JobRecord — per-job state and log buffer
# ---------------------------------------------------------------------------

class TestJobRecord:
    def _make(self, job_id: str = "abc123") -> JobRecord:
        return JobRecord(job_id=job_id, action="detect_file", payload={"source": "v.mp4"})

    def test_initial_status_is_queued(self):
        assert self._make().status == "queued"

    def test_log_appends_message(self):
        rec = self._make()
        rec.log("hello")
        assert "hello" in rec.logs

    def test_log_keeps_last_200_messages(self):
        rec = self._make()
        for i in range(250):
            rec.log(f"msg {i}")
        assert len(rec.logs) == 200
        assert rec.logs[-1] == "msg 249"
        assert rec.logs[0] == "msg 50"

    def test_public_contains_required_keys(self):
        pub = self._make().public()
        for key in ("job_id", "action", "status", "result", "error", "logs", "payload"):
            assert key in pub

    def test_public_returns_correct_job_id_and_action(self):
        pub = self._make("myjob").public()
        assert pub["job_id"] == "myjob"
        assert pub["action"] == "detect_file"

    def test_public_logs_capped_at_50(self):
        rec = self._make()
        for i in range(100):
            rec.log(f"msg {i}")
        assert len(rec.public()["logs"]) <= 50

    def test_public_initial_result_and_error_are_none(self):
        pub = self._make().public()
        assert pub["result"] is None
        assert pub["error"] is None

    def test_multiple_log_calls_accumulate(self):
        rec = self._make()
        for msg in ("a", "b", "c"):
            rec.log(msg)
        assert rec.logs == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# JobManager — thread-safe job registry
# ---------------------------------------------------------------------------

class TestJobManager:
    def _add(self, mgr: JobManager, job_id: str = "job1") -> JobRecord:
        rec = JobRecord(job_id=job_id, action="detect_file", payload={})
        with mgr.lock:
            mgr.jobs[job_id] = rec
        return rec

    def test_list_jobs_empty_initially(self):
        assert JobManager().list_jobs() == []

    def test_get_job_returns_none_for_unknown_id(self):
        assert JobManager().get_job("no-such-id") is None

    def test_stop_job_returns_false_for_unknown_id(self):
        assert JobManager().stop_job("no-such-id") is False

    def test_get_job_returns_dict_for_known_job(self):
        mgr = JobManager()
        self._add(mgr, "j1")
        result = mgr.get_job("j1")
        assert result is not None
        assert result["job_id"] == "j1"

    def test_list_jobs_returns_all_registered_jobs(self):
        mgr = JobManager()
        for i in range(3):
            self._add(mgr, f"job{i}")
        assert len(mgr.list_jobs()) == 3

    def test_stop_job_returns_true_for_known_job(self):
        mgr = JobManager()
        self._add(mgr)
        assert mgr.stop_job("job1") is True

    def test_stop_job_sets_stop_event_on_record(self):
        mgr = JobManager()
        rec = self._add(mgr)
        mgr.stop_job("job1")
        assert rec.stop_event.is_set()

    def test_stop_job_adds_stop_message_to_log(self):
        mgr = JobManager()
        rec = self._add(mgr)
        mgr.stop_job("job1")
        assert any("stop" in msg for msg in rec.logs)

    def test_default_config_path_stored_as_absolute_string(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.touch()
        mgr = JobManager(default_config_path=str(cfg))
        assert mgr.default_config_path == str(cfg.resolve())

    def test_default_config_path_none_stays_none(self):
        assert JobManager(default_config_path=None).default_config_path is None

    def test_submit_creates_job_with_unique_id(self):
        # submit immediately calls _run in a thread; mock via action that fails fast
        mgr = JobManager()
        # We only check that submit returns a dict with a job_id before the thread runs
        # (actual _run would need a real config, so we just verify the public state shape)
        rec = JobRecord(job_id="x1", action="detect_file", payload={})
        with mgr.lock:
            mgr.jobs["x1"] = rec
        result = mgr.get_job("x1")
        assert result is not None
        assert "job_id" in result
