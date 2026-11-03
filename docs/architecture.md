# 架构与接口

王者荣耀自动斗地主机器人（Doudizhu Lab）通过USB采集安卓游戏画面，识别公开局面后由本地模型决策，再经输入闸门操作手机并核对画面反馈。录像、回放和评测为自动对局提供复现与验证。

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

真机采集使用`CaptureProcess`启动独立录像进程，完整编码包先写入分段MKV，再通过有界通道发布分析副本。健康报告与处置命令独立于视频管道和SQLite；主程序阻塞或退出时，录像进程仍能固定故障、尾录和封口。`StopReason`区分用户中止、正常完成和故障，`RecordingResult`返回摘要及实际回放路径。自动会话先停止处理并关闭数据库，再由留存账本清理正常资料；故障、正式验收和显式录像保留复现所需内容。

主要数据流：视频帧 → 同帧观察 → 跟踪与运行状态 → 不可变决策快照 → 模型决策 → 逐次选牌／核对／提交 → 新画面确认。意图、输入返回、确认事件分别记账。

公共Python接口从`domain`导出`Frame`、`TableObservation`、`GameState`、`GameEvent`、`GameTracker`、`ObservationSnapshot`、`Decision`和`Profile`。旧`ddz_device.*`内部导入已移除，命令行入口保留兼容。

模型通过`ModelStatus`发布加载生命周期，连续任务通过`ContinuousProgress`发布只读进度，窗口汇总为`DashboardState`绘制面板，不解析拼接文案。流程持有设备锁时，界面读取缓存进度，不阻塞GUI心跳。

`FlowRecognizer.observe(room=...)`保留旧参数但忽略场次限制；大厅只由通过文字、位置和底色校验的“开始匹配”按钮确认。`FlowObservation.mode_verified`仅兼容旧调用和记录，不参与运行判断。连续协调器以仅存于本任务的`lobby_confirmed`记录有效大厅入口，`startup_entry`记录首次确认的入口，`entry_confirmed`统一判断大厅或结算续局授权。兼容取值`startup_phase="waiting_lobby"`同时表示等待大厅或结算续局入口；结算页只在续局命令取得页面推进反馈后进入`ready`，关闭战绩展示或点击返回大厅本身不授权出牌。`startup_wait_finished`账本记录入口来源；旧局不登记完成，下一局仍须建立完整开局。两帧稳定、单次提交和提交后证据确认保持原逻辑。明确切场提示与普通确认按钮可自动继续，付费或健康提示仍阻断。`Profile.flow_room`读取和保存统一为“当前”。当前第七版配置仅以无业务上限的正整数`max_games`管理任务正常结束，不保存`time_limit`；`ContinuousProgress`与`DashboardState`只发布已用秒数，进度条使用固定范围的整数比例。旧配置首次保存保留原版本备份。

`create_agent(root, backend="resnet")`按明确偏好创建代理，`Profile.play_model`保存选择；旧`Agent(folder)`仍默认为WP。ResNet编码及网络独立保存，界面进程不导入PyTorch。`ModelStatus`和`DashboardState`包含模型后端、显示名称和权重编号；`Decision.strategy`区分`resnet`、`wp`、`rules`，两种模型执行前均经过完整公开局面校验。切换由协调线程串行关闭旧代理、预热新代理，界面立即清除旧建议，迟到结果不得发布。

停止先调用输入闸门的`suspend`锁住重新授权，并通过`request_stop`非阻塞登记意图；后台再关闭账本和录像。只有显式开始、恢复或隔离模拟控制调用`allow`解除停止锁。准备中取消仍必须回收迟到的录像资源。

不要让领域层创建设备对象、让回放持有真实输入接口，或让工作线程持有QObject。动作取消不能等待模型或USB返回才能生效。
