"""InferenceEngine 격리 테스트 — 실제 워커 스레드를 돌리되 NPU 는 열지 않는다."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sys
from threading import Event
import time
import unittest

import numpy as np

from dave_face.pipeline import Reject
from dave_face.web import MODEL, ApiError, InferenceEngine

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class FakePipeline:
    def __init__(self, fail_open=False):
        self.fail_open = fail_open
        self.fail = False
        self.calls = 0
        self.started = Event()
        self.release = Event()
        self.release.set()
        self.closed = Event()

    @contextmanager
    def factory(self):
        if self.fail_open:
            self.closed.set()
            raise RuntimeError("device in use")
        try:
            yield self
        finally:
            self.closed.set()

    def analyze(self, frame):
        self.calls += 1
        self.started.set()
        assert self.release.wait(3), "test did not release analysis"
        if self.fail:
            raise RuntimeError("secret backend details")
        return Reject("no_face")


class EngineTests(unittest.TestCase):
    def make_engine(self, pipeline=None, **kwargs):
        pipeline = pipeline or FakePipeline()
        engine = InferenceEngine(session_factory=pipeline.factory, **kwargs)
        self.addCleanup(engine.close)
        deadline = time.monotonic() + 2
        while engine.status()["state"] == "loading" and time.monotonic() < deadline:
            pipeline.started.wait(0.005)
        return engine, pipeline

    def wait_state(self, engine, state):
        deadline = time.monotonic() + 1
        while engine.status()["state"] != state and time.monotonic() < deadline:
            Event().wait(0.005)
        self.assertEqual(engine.status()["state"], state)

    def test_ready_analyze_close(self):
        engine, pipeline = self.make_engine()
        self.assertEqual(engine.status(), {"state": "ready", "model": MODEL})
        self.assertEqual(engine.analyze(FRAME), Reject("no_face"))
        self.assertEqual(engine.status()["state"], "ready")
        engine.close()
        self.assertTrue(pipeline.closed.wait(1))
        self.assertEqual(engine.status()["state"], "stopped")
        with self.assertRaises(ApiError) as failure:
            engine.analyze(FRAME)
        self.assertEqual(failure.exception.status, 503)

    def test_second_frame_waits_for_the_slot(self):
        engine, pipeline = self.make_engine(wait=2)
        pipeline.release.clear()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(engine.analyze, FRAME)
            self.assertTrue(pipeline.started.wait(1))
            second = pool.submit(engine.analyze, FRAME)
            pipeline.closed.wait(0.05)
            self.assertFalse(second.done())
            pipeline.release.set()
            self.assertEqual(first.result(2), Reject("no_face"))
            self.assertEqual(second.result(2), Reject("no_face"))
        self.assertEqual(pipeline.calls, 2)

    def test_busy_beyond_wait_is_rejected(self):
        engine, pipeline = self.make_engine(wait=0.05)
        pipeline.release.clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(engine.analyze, FRAME)
            self.assertTrue(pipeline.started.wait(1))
            with self.assertRaises(ApiError) as failure:
                engine.analyze(FRAME)
            self.assertEqual((failure.exception.status, failure.exception.payload["error"]), (503, "busy"))
            pipeline.release.set()
            first.result(2)
        self.assertEqual(pipeline.calls, 1)

    def test_failure_returns_to_ready_without_leaking_details(self):
        engine, pipeline = self.make_engine()
        pipeline.fail = True
        with self.assertLogs("dave_face.web", level="ERROR") as logs:
            with self.assertRaises(ApiError) as failure:
                engine.analyze(FRAME)
        self.assertEqual((failure.exception.status, failure.exception.payload), (500, {"error": "inference"}))
        self.assertNotIn("secret", " ".join(logs.output))
        self.assertEqual(engine.status()["state"], "ready")
        pipeline.fail = False
        self.assertEqual(engine.analyze(FRAME), Reject("no_face"))

    def test_open_failure_means_error_state(self):
        with self.assertLogs("dave_face.web", level="ERROR"):
            engine, _ = self.make_engine(FakePipeline(fail_open=True))
            self.wait_state(engine, "error")
        with self.assertRaises(ApiError) as failure:
            engine.analyze(FRAME)
        self.assertEqual((failure.exception.status, failure.exception.payload["error"]), (503, "unavailable"))

    def test_timeout_keeps_worker_busy_until_it_finishes(self):
        engine, pipeline = self.make_engine(timeout=0.03, wait=0.01)
        pipeline.release.clear()
        try:
            with self.assertRaises(ApiError) as failure:
                engine.analyze(FRAME)
            self.assertEqual(failure.exception.status, 504)
            self.assertEqual(engine.status()["state"], "busy")
            with self.assertRaises(ApiError) as busy:
                engine.analyze(FRAME)
            self.assertEqual((busy.exception.status, busy.exception.payload["error"]), (503, "busy"))
        finally:
            pipeline.release.set()
        self.wait_state(engine, "ready")
        self.assertEqual(pipeline.calls, 1)

    def test_sdk_is_never_imported_by_tests(self):
        self.assertEqual([m for m in sys.modules if m.startswith("hailo_platform")], [])


if __name__ == "__main__":
    unittest.main()
