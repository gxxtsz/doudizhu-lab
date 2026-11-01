# 第三方来源与许可

原创代码采用Apache-2.0。第三方部分保留各自许可证、来源与修改记录；模型权重不进入Git、wheel或源码包，也不宣称由本项目训练或具有实测胜率。

| 部分 | 固定来源 | 许可与处理 |
| --- | --- | --- |
| DouZero规则、特征与网络结构 | [kwai/DouZero](https://github.com/kwai/DouZero/tree/718a5c920bf3361e34178a38f3b80458e176b351)，提交`718a5c920bf3361e34178a38f3b80458e176b351` | Apache-2.0；只适配包内导入，未修改规则、特征或网络算法 |
| 三份WP权重 | [社区发布副本](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu/tree/c01062592f362cfb77d6af14a34c2cbcfa614fd2/baselines/douzero_WP)，提交`c01062592f362cfb77d6af14a34c2cbcfa614fd2` | 社区仓库Apache-2.0；下载时保存其许可证，不声称与原官方下载逐字节一致 |
| 叫抢Net2结构与权重 | [FullAuto](https://github.com/Vincentzyx/DouZero_For_HLDDZ_FullAuto/tree/6d20491da216c989d23cf2883498758fb78e9f8d)，提交`6d20491da216c989d23cf2883498758fb78e9f8d` | Apache-2.0；仅移植Net2与计数编码，不引入训练、上游设备操作或未加载网络分支 |
| PP-OCRv5 | PaddleOCR，RapidAI/RapidOCR的`v3.10.0` ONNX转换副本 | Apache-2.0；三个文件的URL和SHA-256固定在视觉资产清单 |
| RapidOCR | 3.10.0 | Apache-2.0；运行依赖与原许可证保留 |
| scrcpy-server | [Genymobile/scrcpy v5.0.1](https://github.com/Genymobile/scrcpy/tree/v5.0.1) | Apache-2.0；显式下载官方发行物，运行只启用视频 |

DouZero和叫抢的原始文件、适配后的文件哈希及来源记录位于`_vendor`中的`provenance.json`。原始许可证不翻译、不改写；说明文档使用中文。YOLO对照实验已移除，不再下载或分发其模型。

PySide6 Essentials、PyTorch、NumPy、OpenCV、PyAV、ONNX Runtime及其依赖由uv按锁文件安装，各自继续适用自身许可，不因本项目许可证而改为Apache-2.0。项目发行包只包含本项目与明确保留的第三方源码／声明，不捆绑这些运行库。

游戏名称、界面、图案和其他游戏素材不属于本项目原创，也不随开源源码分发。项目与游戏、模型及组件发布方不存在隶属或背书关系。
