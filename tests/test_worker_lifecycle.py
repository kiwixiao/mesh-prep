"""Tests for STLClipperApp worker-shutdown helper.

These exercise the pure teardown logic via a fake worker, so they need no
QApplication / display — only that the module imports (PyQt5 present).
"""

from mesh_prep.stl_clipper import STLClipperApp


class _FakeWorker:
    def __init__(self, running, wait_returns=True):
        self._running = running
        self._wait_returns = wait_returns
        self.blocked = False
        self.waited_ms = None

    def isRunning(self):
        return self._running

    def blockSignals(self, flag):
        self.blocked = flag

    def wait(self, ms):
        self.waited_ms = ms
        return self._wait_returns


def test_quiesce_worker_none_is_noop():
    assert STLClipperApp._quiesce_worker(None) is True


def test_quiesce_worker_idle_is_not_touched():
    w = _FakeWorker(running=False)
    assert STLClipperApp._quiesce_worker(w) is True
    assert w.blocked is False          # idle worker left alone
    assert w.waited_ms is None


def test_quiesce_worker_running_blocks_signals_and_waits():
    w = _FakeWorker(running=True)
    result = STLClipperApp._quiesce_worker(w, timeout_ms=1234)
    assert w.blocked is True           # signals detached from the dying window
    assert w.waited_ms == 1234
    assert result is True


def test_quiesce_worker_returns_false_if_still_running_after_timeout():
    w = _FakeWorker(running=True, wait_returns=False)
    assert STLClipperApp._quiesce_worker(w, timeout_ms=10) is False
    assert w.blocked is True
