"""单任务后台执行器；界面一次只投递一项，结果带会话编号用于丢弃过期任务。"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError


class Task:
    """纯 Python 后台任务；不持有 QObject，也不在工作线程调用任何界面方法。"""

    def __init__(self, generation: int, kind: str, action: Callable):
        self.generation, self.kind, self.action = generation, kind, action

    def run(self) -> tuple[object, str]:
        """返回结果及中文错误，由 GUI 定时器主动收取；异常不会越过线程边界。"""
        try:
            return self.action(), ""
        except Exception as exc:
            return None, str(exc)


class SerialWorker:
    """只允许一个在途任务的执行器，拒绝堆积 Future；关闭时等待受超时约束的 ADB 返回。"""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="手机传输")
        self._future: Future | None = None

    def start(self, task: Task) -> Future:
        """执行一个任务；上个任务未结束时抛出异常，作为 GUI busy 门禁之外的第二层保护。"""
        if self.activeThreadCount():
            raise RuntimeError("后台任务尚未结束，不能追加请求")
        self._future = self._executor.submit(task.run)
        return self._future

    def activeThreadCount(self) -> int:
        """返回在途任务数量（0 或 1），包含正在执行而尚未结束的传输。"""
        return int(self._future is not None and not self._future.done())

    def maxThreadCount(self) -> int:
        """固定工作线程上限，供运行统计使用。"""
        return 1

    def waitForDone(self, timeout_ms: int) -> bool:
        """最多等待指定毫秒，不向线程发送强制终止；设备调用自身还有独立超时。"""
        if self._future is not None:
            try:
                self._future.result(timeout=timeout_ms / 1000)
            except TimeoutError:
                return False
        return True

    def close(self) -> None:
        """停止接收任务，等待当前任务结束，释放 Python 工作线程。"""
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._future = None

    def release_completed(self) -> None:
        """由 GUI 线程释放已完成 Future，切断任务闭包与窗口之间的引用环。"""
        if self._future is not None and self._future.done():
            self._future = None
