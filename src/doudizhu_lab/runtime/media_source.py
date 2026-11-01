"""编码媒体源协议；接收与解码流水线不必拥有设备输入客户端。"""

from collections.abc import Iterator
from typing import Protocol

from doudizhu_lab.devices.video import VideoPacket


class VideoSource(Protocol):
    """录制线程独占start/packets；close允许收尾线程调用，以唤醒阻塞读取。"""

    def start(self) -> None:
        """启动本对象的编码源；准备失败应上抛，不自动改用另一个来源。"""
        ...

    def packets(self) -> Iterator[VideoPacket]:
        """依源顺序产出尺寸、配置及媒体包，不静默丢弃完整录像包。"""
        ...

    def close(self) -> None:
        """停止本对象，不清理其他进程、其他USB连接或原录像。"""
        ...
