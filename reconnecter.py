#!/usr/bin/env python3
"""
reconnecter.py — 可靠的连接重连器（熔断 + 退避 + 试探），纯标准库单文件。

状态机：
  CLOSED    正常模式：直接发起真实连接；连续失败达到阈值进入 OPEN。
  OPEN      退避模式：不发起真实连接，connect() 立即快速失败（CircuitOpenError）；
            退避时间到期后进入 HALF_OPEN。
  HALF_OPEN 试探模式：只允许一个“试探连接”在飞（其它请求快速失败）；
            试探成功 -> CLOSED；试探失败 -> 重新 OPEN（退避时长指数增长，封顶）。

默认参数及理由：
  failure_threshold=3      偶发抖动（1~2 次）不熔断，3 次连续失败大概率是对端真挂了。
  backoff_seconds=1.0      初始退避 1s：足够让瞬时拥塞缓一口气，又不至于让用户久等。
  backoff_multiplier=2.0   指数退避：持续故障时迅速降低对服务器的压力。
  max_backoff_seconds=30s  退避上限：避免长时间故障后恢复过慢，30s 是常见的可用性折中。
  max_probe_attempts=5     连续 5 轮试探都失败，说明故障是长时的，重置退避基数重新计，
                           防止试探计数无限增长（退避仍封顶在 max_backoff_seconds）。

并发正确性：
  所有状态迁移都在同一把锁内完成；OPEN 判断与“放行一个试探”是原子的，
  因此状态切换瞬间不会有漏网请求打到服务器，HALF_OPEN 也绝不会同时发出两个试探。
"""

from __future__ import annotations

import threading
import time
import unittest


class CircuitOpenError(Exception):
    """退避/试探期间快速失败抛出的异常（此时并未真正发起连接）。"""


class Reconnecter:
    CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"

    def __init__(
        self,
        connector,
        failure_threshold: int = 3,
        backoff_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: float = 30.0,
        max_probe_attempts: int = 5,
        clock=time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._connector = connector
        self._failure_threshold = failure_threshold
        self._base_backoff = backoff_seconds
        self._multiplier = backoff_multiplier
        self._max_backoff = max_backoff_seconds
        self._max_probe_attempts = max_probe_attempts
        self._clock = clock

        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._current_backoff = backoff_seconds
        self._open_until = 0.0
        self._probe_in_flight = False
        self._probe_round_failures = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def connect(self):
        """获取一个连接。退避期间快速失败；试探期间只放行一个真实连接。"""
        with self._lock:
            now = self._clock()
            if self._state == self.OPEN:
                if now < self._open_until:
                    raise CircuitOpenError(
                        f"circuit OPEN, retry in {self._open_until - now:.3f}s"
                    )
                # 退避到期：原子地切入试探模式（同一锁内完成，无漏网请求）
                self._state = self.HALF_OPEN
                self._probe_in_flight = False
            if self._state == self.HALF_OPEN:
                if self._probe_in_flight:
                    raise CircuitOpenError("probe already in flight")
                self._probe_in_flight = True  # 只有这一个线程被放行去试探
                probing = True
            else:
                probing = False

        # 真实连接放在锁外执行，避免慢连接阻塞状态判断
        try:
            conn = self._connector()
        except Exception:
            if probing:
                self._on_probe_failure()
            else:
                self._on_closed_failure()
            raise

        if probing:
            self._on_probe_success()
        else:
            self._on_closed_success()
        return conn

    # ---- 以下回调都只在持锁时修改状态 ----

    def _on_closed_success(self):
        with self._lock:
            self._consecutive_failures = 0

    def _on_closed_failure(self):
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._enter_open_locked()

    def _on_probe_success(self):
        with self._lock:
            self._state = self.CLOSED
            self._probe_in_flight = False
            self._consecutive_failures = 0
            self._probe_round_failures = 0
            self._current_backoff = self._base_backoff

    def _on_probe_failure(self):
        with self._lock:
            self._probe_in_flight = False
            self._probe_round_failures += 1
            if self._probe_round_failures >= self._max_probe_attempts:
                # 长时故障：重置试探轮次与退避基数，避免计数无限增长
                self._probe_round_failures = 0
                self._current_backoff = self._base_backoff
            self._enter_open_locked()

    def _enter_open_locked(self):
        self._state = self.OPEN
        self._open_until = self._clock() + self._current_backoff
        self._current_backoff = min(
            self._current_backoff * self._multiplier, self._max_backoff
        )


# ---------------------------------------------------------------------------
# 可控 Mock 连接器：按预设脚本依次成功（返回连接对象）或失败（抛异常）
# ---------------------------------------------------------------------------

class MockConnector:
    """script 中每个元素：'ok' 表示成功，'fail' 表示失败；脚本用完后重复最后一项。"""

    def __init__(self, script, latency: float = 0.0):
        self._script = list(script)
        self._latency = latency
        self._lock = threading.Lock()
        self._calls = 0

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def __call__(self):
        with self._lock:
            idx = min(self._calls, len(self._script) - 1)
            outcome = self._script[idx]
            self._calls += 1
        if self._latency:
            time.sleep(self._latency)
        if outcome == "ok":
            return f"mock-conn#{self._calls}"
        raise ConnectionError("mock: connection refused (scripted)")


class FakeClock:
    """可手动推进的时钟，测试退避窗口时不用真的 sleep。"""

    def __init__(self):
        self._now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float):
        with self._lock:
            self._now += seconds


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class ReconnecterTest(unittest.TestCase):
    def make(self, script, **kw):
        clock = FakeClock()
        mock = MockConnector(script)
        r = Reconnecter(mock, clock=clock, **kw)
        return r, mock, clock

    def test_consecutive_failures_enter_backoff(self):
        r, mock, _ = self.make(["fail"] * 10, failure_threshold=3)
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                r.connect()
        self.assertEqual(r.state, Reconnecter.OPEN)
        self.assertEqual(mock.calls, 3)

    def test_fast_fail_during_backoff(self):
        r, mock, clock = self.make(["fail"] * 10, failure_threshold=2,
                                   backoff_seconds=5.0)
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                r.connect()
        calls_before = mock.calls
        clock.advance(1.0)  # 仍在退避窗口内
        for _ in range(5):
            with self.assertRaises(CircuitOpenError):
                r.connect()
        self.assertEqual(mock.calls, calls_before)  # 退避期零真实连接

    def test_probe_success_recovers(self):
        r, mock, clock = self.make(
            ["fail", "fail", "ok", "ok"], failure_threshold=2, backoff_seconds=2.0)
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                r.connect()
        clock.advance(2.0)  # 退避到期 -> 下次 connect 是试探
        conn = r.connect()
        self.assertTrue(conn.startswith("mock-conn"))
        self.assertEqual(r.state, Reconnecter.CLOSED)
        self.assertTrue(r.connect().startswith("mock-conn"))  # 恢复正常

    def test_probe_failure_reopens_with_escalated_backoff(self):
        r, mock, clock = self.make(
            ["fail"] * 10, failure_threshold=2, backoff_seconds=2.0,
            backoff_multiplier=2.0)
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                r.connect()
        clock.advance(2.0)
        with self.assertRaises(ConnectionError):  # 第一次试探失败
            r.connect()
        self.assertEqual(r.state, Reconnecter.OPEN)
        clock.advance(2.0)  # 旧退避窗口，但本次退避已升级为 4s
        with self.assertRaises(CircuitOpenError):
            r.connect()
        clock.advance(2.0)  # 累计 4s，退避到期
        with self.assertRaises(ConnectionError):  # 第二次试探仍失败
            r.connect()
        self.assertEqual(mock.calls, 4)  # 2 次正常失败 + 2 次试探

    def test_single_probe_under_concurrency(self):
        """HALF_OPEN 下多线程同时涌入：只允许一个试探连接真正发出。"""
        clock = FakeClock()
        mock = MockConnector(["fail", "fail", "ok"], latency=0.05)
        r = Reconnecter(mock, failure_threshold=2, backoff_seconds=1.0, clock=clock)
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                r.connect()
        clock.advance(1.0)

        barrier = threading.Barrier(8)
        results, errors = [], []

        def worker():
            barrier.wait()
            try:
                results.append(r.connect())
            except CircuitOpenError as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 1)          # 只有一个试探成功
        self.assertEqual(len(errors), 7)           # 其余快速失败
        self.assertEqual(mock.calls, 3)            # 真实连接总数 = 2 失败 + 1 试探
        self.assertEqual(r.state, Reconnecter.CLOSED)

    def test_no_leak_at_open_transition(self):
        """状态切到 OPEN 的瞬间：切换前已放行的请求照常完成，
        切换后到达的请求一律快速失败，绝不再打到服务器。"""
        clock = FakeClock()
        entered_3rd = threading.Event()
        release_3rd = threading.Event()
        mock = MockConnector(["fail"] * 100)
        orig_call = mock.__call__

        call_count = [0]

        def blocking_call():
            call_count[0] += 1
            if call_count[0] == 3:       # 第 3 次调用先挂起，模拟慢故障
                entered_3rd.set()
                release_3rd.wait(5)
            return orig_call()

        r = Reconnecter(blocking_call, failure_threshold=3, clock=clock)
        # 前两次失败（同步）
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                r.connect()
        # 第 3 次失败在线程中挂起 -> 此时第 4 个请求到达，仍属 CLOSED，应被放行
        def run3():
            with self.assertRaises(ConnectionError):
                r.connect()
        t3 = threading.Thread(target=run3)
        t3.start()
        entered_3rd.wait(5)
        with self.assertRaises(ConnectionError):   # 第 4 次：切换前到达，正常放行
            r.connect()
        release_3rd.set()
        t3.join()
        self.assertEqual(r.state, Reconnecter.OPEN)  # 第 3 次失败落账后熔断
        calls_at_open = mock.calls
        # 切换瞬间之后涌入的并发请求：全部快速失败，零真实连接
        barrier = threading.Barrier(8)
        fast_errors = []

        def worker():
            barrier.wait()
            try:
                r.connect()
            except CircuitOpenError as e:
                fast_errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(fast_errors), 8)
        self.assertEqual(mock.calls, calls_at_open)  # 切换后零漏网


# ---------------------------------------------------------------------------
# 演示：python3 reconnecter.py 运行测试；python3 reconnecter.py demo 看输入输出示例
# ---------------------------------------------------------------------------

def demo():
    clock = FakeClock()
    mock = MockConnector(["fail", "fail", "fail", "ok", "ok"])
    r = Reconnecter(mock, failure_threshold=3, backoff_seconds=2.0, clock=clock)

    def attempt(label):
        try:
            conn = r.connect()
            print(f"{label}: 成功 -> {conn} (state={r.state})")
        except CircuitOpenError as e:
            print(f"{label}: 快速失败 CircuitOpenError({e}) (state={r.state})")
        except ConnectionError as e:
            print(f"{label}: 真实连接失败 {e} (state={r.state})")

    attempt("第1次")            # 真实失败 1
    attempt("第2次")            # 真实失败 2
    attempt("第3次")            # 真实失败 3 -> 达到阈值，进入 OPEN
    attempt("退避期内")          # 快速失败，不打服务器
    clock.advance(2.0)
    attempt("退避到期(试探)")    # 试探成功 -> CLOSED
    attempt("恢复后")            # 正常连接
    print(f"\n服务器实际收到的连接请求数: {mock.calls}（应用层共发起 6 次）")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
