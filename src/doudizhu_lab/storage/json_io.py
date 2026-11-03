"""原子 JSON 发布和有限文件重试；本模块不操作设备，也不吞掉持久化失败。"""

from __future__ import annotations

import errno
import json
from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep
from typing import TypeVar
from uuid import uuid4

T = TypeVar("T")


def temporary_access_error(error: BaseException) -> bool:
    """识别可短暂重试的访问冲突，包括仅提供 errno=13 的 Windows Python 异常。

    磁盘空间不足、路径错误等其他 I/O 错误返回 False，由调用方立即报告实际故障。
    """
    return isinstance(error, PermissionError) and (
        error.errno == errno.EACCES or getattr(error, "winerror", None) in (5, 32, 33)
    )


def retry_file_io(
    operation: Callable[[], T],
    *,
    invalid_json: bool = False,
    on_retry: Callable[[BaseException], None] | None = None,
) -> T:
    """在调用线程中重试短暂文件冲突，重试总预算为二百毫秒，耗尽后原样抛出异常。

    operation 必须自行关闭每次打开的文件。invalid_json 只供运行时消息读取使用，
    允许重试暂不可解析的 JSON；不存在的文件不等待，由首次报告的预热规则处理。
    on_retry 只记录诊断统计，不能取得录像锁或发送设备输入。
    """
    deadline, delay = monotonic() + 0.2, 0.01
    while True:
        try:
            return operation()
        except (OSError, ValueError) as exc:
            retryable = temporary_access_error(exc) or (
                invalid_json and isinstance(exc, (ValueError, UnicodeError))
            )
            if not retryable or monotonic() >= deadline:
                raise
            if on_retry:
                on_retry(exc)
            sleep(min(delay, max(0, deadline - monotonic())))
            delay = min(delay * 2, 0.04)


def atomic_json(
    path: Path, value: object, *, on_retry: Callable[[BaseException], None] | None = None
) -> None:
    """在同一目录原子发布中文 JSON；重试耗尽仍抛错，留存账本不得忽略此错误。

    唯一临时文件避免并发发布互相覆盖。只有 replace 成功才算发布完成，失败不会
    修改上一份有效内容。调用方负责父目录存在及线程归属；本接口不发送设备输入。
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2)

    def publish() -> None:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    try:
        retry_file_io(publish, on_retry=on_retry)
    finally:
        temporary.unlink(missing_ok=True)
