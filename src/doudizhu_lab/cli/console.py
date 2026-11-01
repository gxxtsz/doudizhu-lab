"""统一Windows终端与重定向输出的中文编码。"""

import sys


def configure_console() -> None:
    """启动时在主线程设定标准输出／错误为UTF-8；不影响模型子进程JSON输入协议。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
