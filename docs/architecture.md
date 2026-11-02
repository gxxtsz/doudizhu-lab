# 架构与接口

采用src布局，Python包为`doudizhu_lab`。主窗口持有Qt生命周期；后台只发布同帧不可变快照，不直接访问Qt组件。

| 模块 | 职责 |
| --- | --- |
| `domain` | 牌点、区域、观察、状态、事件、快照与纯配置契约 |
| `devices` | 指定序列号的ADB、固定视频协议与媒体包 |
| `vision` | 模板、牌面、标记、手牌OCR、流程及可点击对象识别 |
| `agents` | 模型准备与CPU子进程、特征编码、规则策略和就绪判断 |
| `runtime` | 解码分析调度、单局执行、连续流程、独占输入与停止 |
| `storage` | 配置、SQLite、证据、录像回放、快照、修订和任务账本 |
| `ui` | 工作台组件、结构化状态、必要设置、只读画布、基础回放纠错及模拟窗口 |
| `cli` | 参数解析、桌面与诊断入口 |
| `evaluation`、`simulation` | 只读评测、故障注入与隔离模拟设备 |
| `_vendor` | 固定版本第三方代码及其原始许可证 |

领域状态不导入Qt、ADB、视觉实现或推理进程。`TableObservation`在领域层定义，视觉实现负责产生它，跟踪器只消费可见证据。默认叫抢设置属于纯配置模块，读取配置不加载模型。

编码媒体源通过`VideoSource`协议注入。原速回放不创建ADB或输入客户端，不需要手机服务器；它发布的画面transport为replay，真实输入闸门会拒绝。真机流水线仍必须通过固定scrcpy服务器校验。历史审计使用`SessionStore(..., readonly=True)`，不会创建表或修改数据库模式。

主要数据流：视频帧 → 同帧观察 → 跟踪与运行状态 → 不可变决策快照 → 模型决策 → 逐次选牌／核对／提交 → 新画面确认。意图、输入返回、确认事件分别记账。

公共Python接口从`domain`导出`Frame`、`TableObservation`、`GameState`、`GameEvent`、`GameTracker`、`ObservationSnapshot`、`Decision`和`Profile`。旧`ddz_device.*`内部导入已移除，命令行入口保留兼容。

模型通过`ModelStatus`发布加载生命周期，连续任务通过`ContinuousProgress`发布只读进度，窗口汇总为`DashboardState`绘制面板，不解析拼接文案。流程持有设备锁时，界面读取缓存进度，不阻塞GUI心跳。

停止先调用输入闸门的`suspend`锁住重新授权，并通过`request_stop`非阻塞登记意图；后台再关闭账本和录像。只有显式开始、恢复或隔离模拟控制调用`allow`解除停止锁。准备中取消仍必须回收迟到的录像资源。

不要让领域层创建设备对象、让回放持有真实输入接口，或让工作线程持有QObject。动作取消不能等待模型或USB返回才能生效。
