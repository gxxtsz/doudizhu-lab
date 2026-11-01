# 贡献指南

界面、文档、日志和新增接口注释使用中文；许可证保留原文。先说明具体问题与预期行为，再提交范围清晰的修改。策略、识别和执行变更分别验证，不以命令成功替代动作确认。

```powershell
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not local_data and not model" -q
uv build
uv run python tools/check_distribution.py
```

公共测试只用合成图像、临时目录和模拟设备。未附带的本机样本与模型必须明确跳过，不把跳过算通过。可选本机回归：

```powershell
uv sync --frozen --extra vision
uv run pytest tests --local-data-dir="你的既有数据目录" --run-model-tests -q
```

本机模型放在工程数据目录；历史样本只读，不复制到仓库。真实手机操作另行约定，不在pytest或CI隐式连接手机。

新增公共接口补充类型标注及中文说明：参数、返回值、异常、线程归属、是否可能发送输入。保持领域状态不依赖设备、界面和模型实现。修改快照与修订逻辑时验证关闭重开、原始观察哈希及旧格式兼容。

第三方代码修改说明上游版本和差异，更新适配文件哈希，不覆盖上游哈希。新增模型需固定来源、SHA-256和许可证；不提交权重、账户配置或来源不明的训练数据。

提交前检查`git status --short`、发行包内容和文档链接。不要把本机目录、完整日志、设备号、账号昵称、录像或截图加入PR。
