"""王者荣耀自动斗地主机器人的中文诊断入口：设备检查、组件准备与离线评测。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from doudizhu_lab import __version__
from doudizhu_lab.devices.adb import AdbClient, AdbError
from doudizhu_lab.domain.models import Frame
from doudizhu_lab.evaluation.cards import evaluate
from doudizhu_lab.storage.config import Store, read_rgb
from doudizhu_lab.vision.cards import Recognizer


def main() -> None:
    """解析命令行并执行设备诊断、组件准备或离线评测。

    本函数不接收 Python 参数，选项及路径从进程命令行读取；正常完成返回 None，
    参数解析或诊断失败可通过 SystemExit 返回非零退出码。设备检查只读取状态，
    组件准备可显式联网并写入本机目录，评测按指定路径保存报告；所有子命令均不
    授予手机自动输入权限，也不发送自动点击或出牌命令。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Windows 重定向时错误流也必须使用 UTF-8，避免中文失败原因与正常输出编码混杂。
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="王者荣耀自动斗地主机器人 · Doudizhu Lab：设备诊断、组件准备与离线评测"
    )
    parser.add_argument("--data-dir", type=Path, help="运行数据目录")
    parser.add_argument(
        "--version", action="version", version=f"王者荣耀自动斗地主机器人 · Doudizhu Lab {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="列出设备及授权状态")
    capture = sub.add_parser("capture", help="获取一帧并导出截图及识别 JSON")
    capture.add_argument("--serial", required=True, help="手机序列号，必须明确指定")
    recognize = sub.add_parser("recognize", help="识别本地截图并导出结果")
    recognize.add_argument("image", type=Path, help="截图文件")
    scoring = sub.add_parser("evaluate", help="评测独立人工标注截图")
    scoring.add_argument("manifest", type=Path, help="标注清单 JSON")
    scoring.add_argument("--output", type=Path, help="结果 JSON 的保存位置")
    sub.add_parser("setup-video", help="下载并校验固定版本 scrcpy-server 5.0.1")
    setup_model = sub.add_parser("setup-agent", help="下载固定出牌权重并预热验证三种身份；不连接手机")
    setup_model.add_argument(
        "--model", choices=("resnet", "wp"), default="resnet", help="默认准备 ResNet，可单独准备 WP"
    )
    sub.add_parser("setup-bidding", help="下载并核验固定叫抢模型，CPU预热；不训练、不连接手机")
    sub.add_parser("setup-vision", help="下载并校验OCR模型；不训练、不连接手机")
    run = sub.add_parser("evaluate-run", help="只读评测连续任务，模拟与真机来源分别标注")
    run.add_argument("folder", type=Path, help="包含run.sqlite3的任务目录")
    run.add_argument("--output", type=Path, help="报告保存位置")
    run.add_argument("--require-pass", action="store_true", help="未达到任务局数或完整连续运行条件返回非零")
    stop_run = sub.add_parser("stop-run", help="请求本工具正常停止指定连续任务，不向手机发送输入")
    stop_run.add_argument("folder", type=Path)
    stop_run.add_argument("--close-window", action="store_true", help="同时正常关闭所属窗口并收尾录像")
    simulation = sub.add_parser("simulate-continuous", help="真实模型完成6个五局流程模拟；不连接手机")
    simulation.add_argument("--runs", type=int, default=6, help="五局任务数，正式离线检查至少6次")
    simulation.add_argument("--output", type=Path)
    simulation.add_argument("--require-pass", action="store_true")
    simulation.add_argument("--play-model", choices=("resnet", "wp"), help="本次模拟使用的出牌模型")
    simulation.add_argument(
        "--bidding", choices=("disabled", "model"), help="模拟中启用真实叫抢评分，实际叫抢结果决定身份"
    )
    simulation.add_argument(
        "--policy", choices=("strict", "infer", "balanced"), help="指定连续策略，仅影响本次隔离模拟"
    )
    vision = sub.add_parser("evaluate-vision", help="在已有人工标注截图上比较免训练识别，不触发输入")
    vision.add_argument("manifests", type=Path, nargs="+", help="现有人工标注清单")
    vision.add_argument(
        "--backend",
        choices=("template", "hand-ocr"),
        required=True,
        help="识别方法",
    )
    vision.add_argument("--output", type=Path, required=True, help="结果JSON")
    vision.add_argument("--limit", type=int, help="仅冒烟取前N张，不作为通过成绩")
    vision.add_argument("--require-pass", action="store_true", help="牌张指标未通过返回2；不表示真机验收")
    vision_image = sub.add_parser("recognize-vision", help="离线查看现成模型的检测框")
    vision_image.add_argument("image", type=Path)
    vision_image.add_argument(
        "--backend",
        choices=("template", "hand-ocr"),
        required=True,
    )
    vision_image.add_argument("--output", type=Path, required=True)
    bot_eval = sub.add_parser("evaluate-bot", help="旧录像建议或模拟闭环评测，不向手机发送输入")
    bot_eval.add_argument("--mode", choices=("replay", "simulation"), required=True, help="评测来源")
    bot_eval.add_argument("--session", type=Path, help="已有连续录像的分析会话目录")
    bot_eval.add_argument("--scope", type=Path, help="回放评测的显式牌局范围JSON；不进入模型输入")
    bot_eval.add_argument("--games", type=int, default=30, help="模拟局数，正式离线检查至少30局")
    bot_eval.add_argument("--output", type=Path, help="评测报告 JSON")
    bot_eval.add_argument("--require-pass", action="store_true", help="未达离线完成条件返回非零退出码")
    bot_eval.add_argument("--play-model", choices=("resnet", "wp"), help="本次建议或模拟使用的出牌模型")
    sub.add_parser("freeze-acceptance", help="冻结程序、配置和模板，为新批次实时验收建立基准")
    reject = sub.add_parser("reject-acceptance", help="将已失败的正式验收批次登记为开发回归资料")
    reject.add_argument("folder", type=Path, help="已停止的正式验收会话目录")
    reject.add_argument("--reason", required=True, help="失败原因，保留在审计记录中")
    audit = sub.add_parser("audit-video", help="逐帧审计录像帧数、时间轴和截断，保存媒体校验报告")
    audit.add_argument("folder", type=Path, help="会话目录")
    correct = sub.add_parser("apply-corrections", help="将已发布的独立整局标注追加为人工纠错，不改原始观察")
    correct.add_argument("folder", type=Path, help="会话目录")
    correct.add_argument("--labels", type=Path, required=True, help="已发布且来源匹配的独立标注 JSON")
    verification = sub.add_parser("verify-corrections", help="核验关闭重开、撤销恢复和原始观察确定性重放")
    verification.add_argument("folder", type=Path, help="会话目录")
    verification.add_argument("--batch", required=True, help="应用纠错时输出的批次编号")
    session_eval = sub.add_parser("evaluate-session", help="评测会话事件、缺口及分局")
    session_eval.add_argument("folder", type=Path, help="会话目录")
    session_eval.add_argument("--labels", type=Path, help="独立人工标注事件 JSON；缺少时不报告准确率")
    session_eval.add_argument("--output", type=Path, help="评测报告路径")
    session_eval.add_argument(
        "--require-pass", action="store_true", help="正式验收未通过时保存报告并返回非零退出码"
    )
    analyze = sub.add_parser("reanalyze", help="从录像生成独立分析版本，原观察不变")
    analyze.add_argument("folder", type=Path, help="会话目录")
    analyze.add_argument("--fps", type=float, default=10, help="离线采样帧率，默认 10")
    analyze.add_argument("--workers", type=int, default=2, help="离线独立帧分析线程数 1～4；结果顺序保持不变")
    analyze.add_argument(
        "--updated-templates", action="store_true", help="使用当前配置模板目录；默认使用录制快照"
    )
    paced = sub.add_parser("paced-replay", help="旧录像按原速复测生产队列；不连接手机，不作为正式验收")
    paced.add_argument("folder", type=Path, help="已停止的原录像会话目录")
    args = parser.parse_args()
    if args.command == "evaluate-bot" and args.mode == "replay" and args.scope is None:
        parser.error("回放评测必须提供 --scope 范围清单")
    try:
        store = Store(args.data_dir)
        if getattr(args, "play_model", None):
            store.profile.play_model = args.play_model
        recognizer = Recognizer(store.template_dir)
        if args.command == "devices":
            print(
                json.dumps(
                    [asdict(d) for d in AdbClient(store.adb_path).list_devices()],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command in ("capture", "recognize"):
            frame = (
                AdbClient(store.adb_path, args.serial).capture()
                if args.command == "capture"
                else Frame("offline", read_rgb(args.image), monotonic(), args.image.name, 0, "file")
            )
            result = recognizer.recognize(frame, store.profile)
            print(f"已保存：{store.export(frame, result)}")
        elif args.command == "setup-video":
            from doudizhu_lab.devices.video import SERVER_SHA256, setup_video

            print(f"视频组件就绪：{setup_video(store.root)}\nSHA-256：{SERVER_SHA256}")
        elif args.command == "setup-vision":
            from doudizhu_lab.vision.assets import setup_vision

            print(json.dumps(setup_vision(store.root), ensure_ascii=False, indent=2))
        elif args.command == "evaluate-run":
            from doudizhu_lab.storage.runs import evaluate_run

            report = evaluate_run(args.folder, output=args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.require_pass and not report["passed"]:
                parser.exit(2, "连续任务检查未通过；报告已保存\n")
        elif args.command == "stop-run":
            if not (args.folder / "run.sqlite3").is_file():
                raise ValueError("没有找到该连续任务")
            target = args.folder / "停止请求.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({"close_window": args.close_window}), encoding="utf8")
            temporary.replace(target)
            print("已请求停止，等待本工具收尾；已发出的在途命令不能撤回")
        elif args.command == "simulate-continuous":
            from doudizhu_lab.simulation.continuous import simulate_continuous

            if args.runs < 1:
                raise ValueError("模拟任务数必须为正整数")
            if args.policy:
                store.profile.continuous_policy = args.policy
            if args.bidding:
                store.profile.bid_settings["mode"] = args.bidding
            report = simulate_continuous(store, runs=args.runs, output=args.output)
            print(f"连续流程模拟{'通过' if report['passed'] else '未通过'}；真机成绩单独报告")
            if args.require_pass and not report["passed"]:
                parser.exit(2, "连续流程模拟未通过\n")
        elif args.command == "evaluate-vision":
            from doudizhu_lab.evaluation.vision import compare_vision

            if args.limit is not None and args.limit < 1:
                raise ValueError("冒烟数量必须为正整数")
            result = compare_vision(
                store, args.manifests, backend=args.backend, output=args.output, limit=args.limit
            )
            if args.require_pass and not result["rank_metrics_passed"]:
                parser.exit(2, "免训练识别牌张指标未通过\n")
        elif args.command == "recognize-vision":
            from doudizhu_lab.evaluation.vision import export_prediction

            export_prediction(store, args.image, backend=args.backend, output=args.output)
            print(f"离线检测结果已保存：{args.output}")
        elif args.command == "setup-agent":
            from doudizhu_lab.agents.backends import setup_play_model

            print(json.dumps(setup_play_model(store.root, args.model), ensure_ascii=False, indent=2))
        elif args.command == "setup-bidding":
            from doudizhu_lab.agents.bidding import setup_bidding

            print(json.dumps(setup_bidding(store.root), ensure_ascii=False, indent=2))
        elif args.command == "evaluate-bot":
            from doudizhu_lab.simulation.bot import evaluate_bot

            output = args.output or store.root / "bot-evaluation" / f"{args.mode}.json"
            report = evaluate_bot(
                store, mode=args.mode, session=args.session, games=args.games, scope=args.scope
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"离线机器人检查{'通过' if report['passed'] else '未通过'}：{output}；真机操作尚未实测")
            if args.require_pass and not report["passed"]:
                parser.exit(2, "离线机器人评测未达到完成条件\n")
        elif args.command == "freeze-acceptance":
            from doudizhu_lab.storage.session import freeze_release

            print(f"验收基准已保存：{freeze_release(store)}；尚未开始录制")
        elif args.command == "reject-acceptance":
            from doudizhu_lab.storage.session import SessionStore

            session = SessionStore(args.folder)
            try:
                result = session.reject_acceptance(args.reason)
                session.export()
                print(f"该批次已转为开发回归资料：{result['reason']}；原始录像与实时结果保留")
            finally:
                session.close()
        elif args.command == "apply-corrections":
            from doudizhu_lab.storage.corrections import apply_label_corrections

            result = apply_label_corrections(args.folder, args.labels)
            print(f"已追加纠错批次：{result['batch_id']}；修订 {len(result['revisions'])} 条")
        elif args.command == "verify-corrections":
            from doudizhu_lab.storage.corrections import verify_correction_replay

            result = verify_correction_replay(args.folder, args.batch)
            target = args.folder / "纠错复现检查.json"
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已保存：{target}")
            if not result["passed"]:
                parser.exit(2, "纠错或相同观察重放未通过一致性检查\n")
        elif args.command == "audit-video":
            from doudizhu_lab.storage.replay import VideoReplay

            report = VideoReplay(args.folder).audit()
            target = args.folder / "media-audit.json"
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"录像完整性{'通过' if report['passed'] else '未通过'}：{target}")
            if not report["passed"]:
                parser.exit(2, "存在未收尾、截断或解码异常的录像\n")
        elif args.command == "evaluate-session":
            from doudizhu_lab.storage.replay import evaluate_session

            report = evaluate_session(args.folder, args.labels)
            output = args.output or args.folder / "整局评测.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已保存：{output}" + ("" if args.labels else "；未提供独立人工标注，准确率未评测"))
            if args.require_pass and not report["target_passed"]:
                parser.exit(2, "正式验收未通过：" + "；".join(report["unmet"]) + "\n")
        elif args.command == "paced-replay":
            from doudizhu_lab.evaluation.paced_replay import paced_replay

            # 使用当前用户配置；诊断产物单独保存，避免与真实手机采集混淆。
            diagnostic = Store(store.root / "diagnostics/paced-full")
            diagnostic.profile = store.profile
            result = paced_replay(
                args.folder,
                diagnostic,
                progress=lambda s: print(
                    f"原速 {s['source_seconds']:.1f} 秒；解码 {s['decoded']}，分析 {s['analyzed']}，"
                    f"在途 {s['analysis_queue']}，繁忙跳过 {s['skipped']}；{s['folder']}",
                    flush=True,
                ),
            )
            print(f"原速诊断已保存：{result}；非真机实时验收")
        elif args.command == "reanalyze":
            from doudizhu_lab.storage.replay import reanalyze

            if not 0 < args.fps <= 30:
                raise ValueError("离线采样帧率必须在 0～30 之间")
            print(
                f"分析版本：{reanalyze(args.folder, template_root=store.root if args.updated_templates else None, fps=args.fps, workers=args.workers, progress=lambda n: print(f'已分析 {n} 帧', flush=True))}"
            )
        else:
            report = evaluate(args.manifest, recognizer, store.profile)
            output = args.output or store.root / "识别评测.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"样本 {report['samples']} 张；精确率 {report['total']['precision']:.2%}；召回率 {report['total']['recall']:.2%}；目标{'达到' if report['target_passed'] else '未达到'}\n结果：{output}"
            )
    except (AdbError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"操作失败：{exc}\n")


if __name__ == "__main__":
    main()
