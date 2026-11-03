"""显式生成或执行旧录像清理清单；不停止程序、不删除清单以外的资料。"""

import argparse
import json
from pathlib import Path

from doudizhu_lab.cli.console import configure_console
from doudizhu_lab.storage.cleanup import apply_legacy_cleanup, legacy_inventory
from doudizhu_lab.storage.retention import atomic_json


def main() -> None:
    """默认只生成清单；--apply执行已有清单，运行中或路径不合法时返回失败。"""
    configure_console()
    parser = argparse.ArgumentParser(description="旧录像整会话清理，保留全局配置、模板和模型")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        value = apply_legacy_cleanup(args.data_dir, args.manifest)
    else:
        if args.manifest.exists():
            parser.error("清单已存在，请使用新文件名或执行原清单")
        value = legacy_inventory(args.data_dir)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.manifest, value)
    print(
        json.dumps(
            {
                "会话数": len(value["sessions"]),
                "关联任务数": len(value["runs"]),
                "清理字节数": value["freed_bytes"],
                "错误": value["errors"],
                "剩余目录": value.get("remaining", []),
                "受保护资料未变": value.get("protected_unchanged"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if value["errors"] or value.get("remaining") or value.get("protected_unchanged") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
