"""纯界面状态；只传递可展示的数据，不持有 Qt、设备或执行授权。"""

from dataclasses import dataclass


def duration_text(seconds: float) -> str:
    """把非负秒数格式化为时分秒；纯函数，不修改输入，也不访问设备。"""
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


@dataclass(frozen=True)
class DashboardState:
    """GUI 线程的一次只读展示快照，不通过显示文本反推任务或模型状态。

    status 是界面生命周期，model_state 是模型生命周期；两者相互独立。
    局数为 Python 整数，不设业务上限；elapsed 只显示已用秒数，不设任务截止时间。
    局数和用时来自任务快照，未知局面显示占位，不猜测牌点或剩牌数。
    """

    status: str = "idle"
    model_state: str = "unloaded"
    model_error: str = ""
    device_text: str = "未连接设备"
    device_ready: bool = False
    scene: str = "等待开始"
    completed: int = 0
    total: int = 5
    elapsed: float = 0
    notice: str = "模型就绪并连接手机后，即可开始连续游戏。"
    role: str = "—"
    actor: str = "—"
    remaining: tuple[str, str, str] = ("—", "—", "—")
    action: str = "等待新的牌局"
    saved: bool = False
    model_backend: str = "resnet"
    model_name: str = "ResNet"
    model_id: str = ""
