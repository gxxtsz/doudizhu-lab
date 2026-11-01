"""有界实时识别调度；原图独立计算，完成结果仍按媒体顺序交给状态机。"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from doudizhu_lab.domain.models import Frame


class MediaSampler:
    """按照源时间的固定刻度采样，避免 15 FPS 输入被逐帧等待降为 7.5 FPS。

    例如 10 FPS 分析在 15 FPS 视频中交替间隔一帧和两帧，长期不超过配置频率。
    编码会话切换后重新建立刻度；重复或倒序帧不重复分析。
    """

    def __init__(self, fps: float = 10, *, minimum_spacing: bool = False):
        """minimum_spacing用于自动任务：每两次分析至少相隔一个刻度，避免94ms证据对。

        默认保留旧版固定刻度行为。严格间隔在15FPS视频上通常约7.5FPS，仍满足实时
        分析至少5FPS及上限10FPS；不降低跟踪器要求的100ms稳定证据门槛。
        """
        if not 0 < fps <= 10:
            raise ValueError("实时牌桌识别频率必须大于零且不超过 10 FPS")
        self.interval = round(1_000_000 / fps)
        self.minimum_spacing = minimum_spacing
        self.epoch = None
        self.next_pts = None
        self.previous = -1

    def accept(self, frame: Frame) -> bool:
        """仅决定本帧是否位于下一个采样刻度，不改变画面、时间戳或录像内容。"""
        pts = frame.media_pts_us
        if pts is None:
            return False
        if frame.stream_id != self.epoch:
            self.epoch = frame.stream_id
            self.next_pts, self.previous = None, -1
        if pts <= self.previous:
            return False
        self.previous = pts
        if self.next_pts is not None and pts < self.next_pts:
            return False
        self.next_pts = (
            pts + self.interval
            if self.next_pts is None or self.minimum_spacing
            else self.next_pts + ((pts - self.next_pts) // self.interval + 1) * self.interval
        )
        return True


class OrderedAnalyzer:
    """最多保留六个任务，两条独立识别线程，结果严格按提交顺序返回。

    factory 在线程内建立独享的识别器；模板缓存不跨线程写入。积压时拒收新分析帧，
    原录像仍由其他流水线完整保存。take 会释放锁再等待计算，不阻塞解码线程提交。
    """

    def __init__(self, factory, *, workers: int = 2, capacity: int = 6):
        if workers < 1 or capacity < workers:
            raise ValueError("识别队列容量不能小于工作线程数")
        self.factory = factory
        self.capacity = capacity
        self.local = threading.local()
        self.condition = threading.Condition()
        self.pending = deque()
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="实时独立帧识别")

    def _run(self, frame):
        if not hasattr(self.local, "recognize"):
            self.local.recognize = self.factory()
        return frame, self.local.recognize(frame)

    def submit(self, frame: Frame) -> bool:
        """无阻塞提交；返回 False 表示任务已满或停止，调用方应登记跳过的分析帧。"""
        with self.condition:
            if self.closed or len(self.pending) >= self.capacity:
                return False
            self.pending.append(self.executor.submit(self._run, frame))
            self.condition.notify()
            return True

    def take(self, timeout=0.2):
        """等待最早提交的任务；无任务返回 None，识别异常原样交给流水线登记缺口。

        已完成的后帧不能越过尚未完成的前帧，因此状态机始终得到递增源时间。
        """
        with self.condition:
            self.condition.wait_for(lambda: self.pending or self.closed, timeout=timeout)
            if not self.pending:
                return None
            future = self.pending[0]
        result = future.result()
        with self.condition:
            self.pending.popleft()
        return result

    @property
    def size(self):
        """包含正在计算和等待消费的任务数，供界面与验收记录实际积压。"""
        with self.condition:
            return len(self.pending)

    def close(self):
        """停止接收新任务并唤醒消费者；已接收的任务继续处理，不静默丢弃尾部。"""
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def shutdown(self):
        """等待已提交任务收尾；必须在消费者退出或已排空后调用。"""
        self.close()
        self.executor.shutdown(wait=True, cancel_futures=False)
