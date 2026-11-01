# 配置与数据

数据目录依次选择：

1. 显式`--data-dir`参数。
2. `DDZ_DATA_DIR`环境变量。
3. 从源码运行时的工程`data`目录。
4. 安装版的`%LOCALAPPDATA%\DoudizhuLab`。

不根据终端当前目录猜测工程，也不将运行资料写入`site-packages`。其他平台存在XDG路径退路，但目前只声明Windows支持。

```powershell
uv run ddz-lab --data-dir .\data
uv run ddz-check --data-dir .\data devices
```

ADB路径以调用者明确指定和保存的配置为准；未配置时读取`ADB_PATH`，再查找PATH。没有作者电脑固定盘符依赖。所有设备命令必须指定序列号。

## 本机目录

| 目录／文件 | 用途 | 提交到Git |
| --- | --- | --- |
| `settings.json` | 第四版配置与当前标定 | 否 |
| `templates`、`markers`、`controls` | 牌点、标记和按钮模板 | 否 |
| `models`、`tools` | 已校验权重与视频服务器 | 否 |
| `sessions`、`runs` | 录像、观察、修订和任务账本 | 否 |
| `migration` | 本机迁移和验证记录 | 否 |

旧配置按原规则升级，首次保存时备份。程序和内部包改名不改变第四版字段及既有JSON／SQLite格式。示例配置不含真实设备路径或可执行的按钮标定。

启动默认仅建议；运行授权、正在进行的动作和连续执行状态不跨程序重启保存。新配置默认严格模式、固定不叫不抢、始终不加倍、最多5局或30分钟。用户可显式选择平衡模式和模型叫抢；本次迁移的本机配置保留既有偏好。

默认录像总额20 GiB，磁盘空闲低于2 GiB停止录像，不自动删除旧资料。
