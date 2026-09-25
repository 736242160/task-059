"""reconnector.py — 可靠的连接重连器（熔断 + 指数退避 + 单试探），纯标准库单文件。

状态机:
    CLOSED    正常态: 直接发起连接; 连续失败达阈值 -> OPEN
    OPEN      退避态: 不发起真实连接, 立即快速失败 (CircuitOpenError);
              退避计时到期 -> HALF_OPEN
    HALF_OPEN 试探态: 只放行 max_probes 个试探连接, 其余快速失败;
              试探成功 -> CLOSED; 试探失败 -> OPEN (退避时长指数升级, 封顶 max_backoff)

并发安全性: 所有状态读取/切换都在同一把锁内完成,
    - 进入 OPEN 的瞬间, 之后任何 connect() 都在锁内看到 OPEN, 不会漏网打到服务器;
    - HALF_OPEN 下 probes_in_flight 计数在锁内递增, 保证同时最多 max_probes 个试探。

默认参数及理由:
    failure_threshold=3    容忍偶发抖动(1~2 次), 又不至于让故障拖太久才熔断
    base_backoff=1.0 s     首次退避 1s, 给服务器短暂喘息, 对调用方也可接受
    backoff_multiplier=2.0 指数退避, 故障越久打得越少, 避免持续压垮服务器
    max_backoff=30.0 s     退避封顶, 保证恢复延迟有上界, 不会无限指数增长
    max_probes=1           试探只放一个, 防止恢复瞬间惊群(thundering herd)
"""

from __future__ import annotations

import threading
import time
import unittest
from collections import deque


class CircuitOpenError(Exception):
    """退避/试探期的快速失败异常, 不发起真实连接。"""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"circuit open, fast fail, retry after {retry_after:.3f}s")


class Reconnector:
    """封装 connect() 入口的熔断重连器。线程安全。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        connector,
        *,
        failure_threshold: int = 3,
        base_backoff: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_backoff: float = 30.0,
        max_probes: int = 1,
        clock=time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if max_probes < 1:
            raise ValueError("max_probes must be >= 1")
        self._connector = connector
        self._failure_threshold = failure_threshold
        self._base_backoff = base_backoff
        self._multiplier = backoff_multiplier
        self._max_backoff = max_backoff
        self._max_probes = max_probes
        self._clock = clock

        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._backoff = base_backoff
        self._opened_at = 0.0
        self._probes_in_flight = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def connect(self):
        """连接入口。退避期快速失败; 试探期只放行有限个试探连接。"""
        with self._lock:
            now = self._clock()
            if self._state == self.OPEN:
                remaining = self._backoff - (now - self._opened_at)
                if remaining > 0:
                    raise CircuitOpenError(remaining)  # 快速失败, 不打服务器
                self._state = self.HALF_OPEN           # 退避到期, 原子切到试探态
                self._probes_in_flight = 0
            if self._state == self.HALF_OPEN:
                if self._probes_in_flight >= self._max_probes:
                    raise CircuitOpenError(0.0)        # 已有试探在飞, 不再放
                self._probes_in_flight += 1
                is_probe = True
            else:
                is_probe = False
        # 真实连接放在锁外执行, 避免慢连接阻塞状态判断
        try:
            conn = self._connector()
        except Exception:
            self._on_failure(is_probe)
            raise
        self._on_success(is_probe)
        return conn

    def _open(self, escalate: bool):
        # 调用时必须已持有锁: 状态切换与后续 connect() 的判断互斥, 无漏网请求
        if escalate:
            self._backoff = min(self._backoff * self._multiplier, self._max_backoff)
        self._state = self.OPEN
        self._opened_at = self._clock()
        self._consecutive_failures = 0

    def _on_failure(self, is_probe: bool):
        with self._lock:
            if is_probe:
                self._probes_in_flight -= 1
                self._open(escalate=True)              # 试探失败, 退避升级
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_threshold:
                    self._open(escalate=False)

    def _on_success(self, is_probe: bool):
        with self._lock:
            if is_probe:
                self._probes_in_flight -= 1
            self._state = self.CLOSED                  # 试探/普通成功都恢复正常
            self._consecutive_failures = 0
            self._backoff = self._base_backoff


# ---------------------------------------------------------------------------
# 可控 mock: 按预设脚本成功或失败的连接器
# ---------------------------------------------------------------------------

class ScriptedConnector:
    """script 中每个元素: True -> 成功; False 或 Exception 实例 -> 失败。
    脚本用完后默认成功。calls 记录真实发起连接的次数(验证快速失败没打服务器)。"""

    def __init__(self, script=()):
        self._script = deque(script)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
            outcome = self._script.popleft() if self._script else True
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is False:
            raise ConnectionError("scripted connection failure")
        return FakeConnection(self.calls)


class FakeConnection:
    def __init__(self, conn_id: int):
        self.conn_id = conn_id

    def __repr__(self):
        return f"<FakeConnection #{self.conn_id}>"


class FakeClock:
    """可手动推进的时钟, 让退避测试无需真实等待。"""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt: float):
        self.t += dt


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class ReconnectorTest(unittest.TestCase):
    def make(self, script, **kw):
        clock = FakeClock()
        connector = ScriptedConnector(script)
        rc = Reconnector(connector, clock=clock, **kw)
        return rc, connector, clock

    def test_consecutive_failures_enter_backoff(self):
        rc, connector, _ = self.make([False] * 3, failure_threshold=3)
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                rc.connect()
        self.assertEqual(rc.state, Reconnector.OPEN)
        self.assertEqual(connector.calls, 3)

    def test_fast_fail_during_backoff(self):
        rc, connector, _ = self.make([False] * 3, failure_threshold=3, base_backoff=10.0)
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                rc.connect()
        # 退避期内连续请求: 全部快速失败, 不再真实连接
        for _ in range(5):
            with self.assertRaises(CircuitOpenError):
                rc.connect()
        self.assertEqual(connector.calls, 3)  # 服务器零额外压力

    def test_probe_success_recovers(self):
        rc, connector, clock = self.make(
            [False] * 3 + [True, True], failure_threshold=3, base_backoff=5.0)
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                rc.connect()
        self.assertEqual(rc.state, Reconnector.OPEN)
        clock.advance(5.0)                      # 退避到期
        conn = rc.connect()                     # 试探成功
        self.assertIsInstance(conn, FakeConnection)
        self.assertEqual(rc.state, Reconnector.CLOSED)
        self.assertIsInstance(rc.connect(), FakeConnection)  # 恢复后正常连接
        self.assertEqual(connector.calls, 5)

    def test_probe_failure_backs_off_again(self):
        rc, connector, clock = self.make(
            [False] * 4 + [True], failure_threshold=3, base_backoff=5.0)
        for _ in range(3):
            with self.assertRaises(ConnectionError):
                rc.connect()
        clock.advance(5.0)
        with self.assertRaises(ConnectionError):  # 第 4 次: 试探失败
            rc.connect()
        self.assertEqual(rc.state, Reconnector.OPEN)
        self.assertEqual(connector.calls, 4)
        # 退避升级为 10s: 第 6 秒仍快速失败
        clock.advance(6.0)
        with self.assertRaises(CircuitOpenError):
            rc.connect()
        self.assertEqual(connector.calls, 4)
        # 熬过升级后的退避, 再试探成功即恢复
        clock.advance(4.0)
        self.assertIsInstance(rc.connect(), FakeConnection)
        self.assertEqual(rc.state, Reconnector.CLOSED)

    def test_only_one_probe_in_flight(self):
        """并发卡在试探切换瞬间: 同时只能有一个试探连接。"""
        entered = threading.Event()
        release = threading.Event()
        inflight = {"cur": 0, "max": 0}
        lock = threading.Lock()

        def blocking_probe():
            with lock:
                inflight["cur"] += 1
                inflight["max"] = max(inflight["max"], inflight["cur"])
            entered.set()
            release.wait(2.0)
            with lock:
                inflight["cur"] -= 1
            return FakeConnection(99)

        clock = FakeClock()
        rc = Reconnector(blocking_probe, failure_threshold=1,
                         base_backoff=1.0, clock=clock)
        rc._connector = ScriptedConnector([False])  # 先熔断
        with self.assertRaises(ConnectionError):
            rc.connect()
        rc._connector = blocking_probe
        clock.advance(1.0)  # 退避到期, 下次 connect 进入试探

        results, errors = [], []
        def worker():
            try:
                results.append(rc.connect())
            except CircuitOpenError:
                errors.append("fast-fail")

        probe_thread = threading.Thread(target=worker)
        probe_thread.start()
        entered.wait(2.0)                     # 确认试探已在飞
        others = [threading.Thread(target=worker) for _ in range(5)]
        for t in others:
            t.start()
        for t in others:
            t.join()
        release.set()
        probe_thread.join()

        self.assertEqual(inflight["max"], 1)          # 同时只有一个试探
        self.assertEqual(len(results), 1)             # 只有试探成功
        self.assertEqual(len(errors), 5)              # 其余全部快速失败
        self.assertEqual(rc.state, Reconnector.CLOSED)

    def test_no_leak_at_open_transition(self):
        """刚进入退避的瞬间: 阈值达成后任何并发请求都不能打到服务器。"""
        clock = FakeClock()
        connector = ScriptedConnector([False] * 2)  # 之后默认成功
        rc = Reconnector(connector, failure_threshold=2,
                         base_backoff=60.0, clock=clock)
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                rc.connect()
        # 熔断瞬间起 20 个并发请求, 全部必须快速失败
        errors = []

        def worker():
            try:
                rc.connect()
            except CircuitOpenError:
                errors.append(1)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 20)
        self.assertEqual(connector.calls, 2)  # 无一漏网


# ---------------------------------------------------------------------------
# 输入输出示例
# ---------------------------------------------------------------------------

def demo():
    print("=== demo: 连续失败 -> 退避快速失败 -> 试探失败 -> 试探成功恢复 ===")
    clock = FakeClock()
    # 脚本: 3 连败触发熔断, 第 1 次试探失败, 第 2 次试探成功后恢复
    connector = ScriptedConnector([False, False, False, False, True, True])
    rc = Reconnector(connector, failure_threshold=3, base_backoff=2.0, clock=clock)

    def attempt(tag):
        try:
            conn = rc.connect()
            print(f"t={clock.t:5.1f}s [{tag}] OK        -> {conn} (state={rc.state})")
        except CircuitOpenError as e:
            print(f"t={clock.t:5.1f}s [{tag}] FAST-FAIL -> {e} (state={rc.state})")
        except ConnectionError as e:
            print(f"t={clock.t:5.1f}s [{tag}] FAIL      -> {e} (state={rc.state})")

    for i in range(3):
        attempt(f"req{i+1}")            # 3 次真实连接失败 -> OPEN
    attempt("req4")                     # 退避期: 快速失败, 不打服务器
    clock.advance(2.0)
    attempt("req5")                     # 退避到期: 试探失败 -> 退避升级为 4s
    clock.advance(3.0)
    attempt("req6")                     # 升级退避未到期: 快速失败
    clock.advance(1.0)
    attempt("req7")                     # 试探成功 -> CLOSED
    attempt("req8")                     # 恢复正常
    print(f"真实连接总次数 = {connector.calls} (退避期请求未打到服务器)")


if __name__ == "__main__":
    demo()
    print("\n=== unittest ===")
    unittest.main(argv=[__file__, "-v"], exit=False)
