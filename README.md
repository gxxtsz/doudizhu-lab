# 斗地主实验室 · Doudizhu Lab

通过USB连接安卓真机的斗地主研究工具：画面采集、公开牌面识别、整局记录与回放、人工纠错，以及基于固定DouZero模型的动作闭环。

**实验阶段，启动后台加载本地ResNet模型并只读预览。** 自动操作必须显式点击“开始连续游戏”。当前只完成Windows与特定游戏布局的验证，不能开箱适配所有设备。模型、游戏图片、模板、手机配置和历史录像不随源码分发。

## 功能

- 自动只读预览，独立进程采集并分段保存MKV；正常局及主动中止局清理临时录像，程序异常保留故障资料。
- 五区牌面、阶段、地主、剩牌数、行动与不出提示识别；可选免训练手牌OCR。
- SQLite原始观察、事件、疑点、修订和动作账本；同版源码及模板快照。
- 原始与运行分局通过发牌边界关联，完成计数前独立复核证据，异常局固定媒体供排查。
- DouZero ResNet默认CPU推理，可切换WP；现成叫抢模型、逐张选牌核对、单次提交和画面确认；始终不加倍。
- 连续游戏工作台、必要设置和基础回放纠错；异常局自动隔离，故障工作进程自动重建，保留有效进度直到补足目标局数。所有模式接受可独立复核的确定性推断；平衡模式另外允许明确回合的规则接管。
- 沿用当前大厅或从结算页续局，洗牌、不洗牌及其他场次均不按名称限制入口；出牌仍使用现有标准54张牌规则。
- 无需手机的模拟窗口与离线评测。运行数据保存在本机，不默认上传。

## 快速开始

需要Windows、Python 3.11和[uv](https://docs.astral.sh/uv/)。在本项目目录执行：

```powershell
uv sync --frozen
uv run ddz-lab --version
uv run ddz-lab
```

`ddz-device`保留为兼容入口，诊断命令为`ddz-check`。独立准备模型和视频组件：

```powershell
uv sync --frozen --extra vision
uv run ddz-check setup-video
uv run ddz-check setup-agent
uv run ddz-check setup-bidding
uv run --extra vision ddz-check setup-vision
uv run ddz-bot-demo
```

上述准备命令显式下载固定版本并校验；演示不连接手机。真机还需要ADB、手机授权及已有的牌面模板与按钮标定。日常界面不提供首次标定工具，详见[安装指南](docs/getting-started.md)和[标定配置说明](docs/calibration.md)。

## 文档

| 主题 | 入口 |
| --- | --- |
| 安装、手机连接与常见问题 | [安装与启动](docs/getting-started.md) |
| 数据目录、ADB与配置兼容 | [配置说明](docs/configuration.md) |
| 已有模板、区域和操作布局 | [标定配置](docs/calibration.md) |
| 录像、回放纠错及命令行重算 | [记录与回放](docs/recording-and-replay.md) |
| 模型、推断、连续任务与自动恢复 | [模型与连续运行](docs/agent-and-continuous.md) |
| 模块职责与公共数据接口 | [架构](docs/architecture.md) |
| 固定来源、许可与第三方改动 | [第三方来源](docs/third-party.md) |
| 实测程度与本次检查 | [验证状态](docs/validation.md) |
| 旧项目到0.5版的变化 | [迁移说明](docs/migration.md) |

## 验证程度

重构前的单个真机任务连续完成15局、14次衔接及170次本人动作确认，第16局被游戏健康系统强制休息中断。**真机连续30局验收尚未通过**，实时整局事件识别的正式验收也仍未通过。

15局中138次动作为DouZero WP、32次为规则策略，另有1个事件通过独立证据推断；这些不能合称全程模型出牌，也不代表对战胜率。离线模拟、历史数据回归与真机成绩分别统计。

默认视频上限15 FPS、识别上限10 FPS是调度参数，不是所有设备的实测保证。传统候选分割仍可能受动画、遮挡和叠牌影响；未知提交不盲目重试，健康系统、充值与账户提示不会自动绕过。

## 开发

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not local_data and not model" -q
uv build
uv run python tools/check_distribution.py
```

公共测试使用合成数据和模拟设备，不下载权重、不连接手机。完整可选回归、中文接口规范与提交说明见[贡献指南](CONTRIBUTING.md)。

## 许可

原创代码采用[Apache-2.0](LICENSE)。第三方源码、依赖、模型和游戏素材保留各自权利，详见[NOTICE](NOTICE)和[来源说明](docs/third-party.md)。本项目与游戏或模型发布方没有隶属或背书关系。隐私问题见[安全说明](SECURITY.md)。
