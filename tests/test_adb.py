"""验证传输边界、原始帧解析和手机错误状态，不接触真机。"""

import struct
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from doudizhu_lab.devices.adb import AdbClient, AdbError, RawFormatError, decode_raw


@pytest.mark.parametrize("header", [12, 16])
@pytest.mark.parametrize("fmt,channels", [(1, 4), (2, 4), (3, 3)])
def test_raw_headers_and_color(header, fmt, channels):
    pixels = np.array([[[220, 40, 10, 255], [17, 80, 230, 255]]], np.uint8)[:, :, :channels]
    blob = struct.pack("<III", 2, 1, fmt) + (struct.pack("<I", 2) if header == 16 else b"") + pixels.tobytes()
    decoded = decode_raw(blob)
    assert decoded.tolist() == pixels[:, :, :3].tolist()
    assert decoded.flags.c_contiguous
    assert decoded.flags.owndata


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"abc",
        struct.pack("<III", 0, 2, 1),
        struct.pack("<III", 10001, 1, 1),
        struct.pack("<IIII", 2, 2, 1, 2) + bytes([10, 20, 30, 255]) * 3,
    ],
)
def test_corrupt_raw_never_returns_success(blob):
    with pytest.raises(ValueError):
        decode_raw(blob)


def test_unsupported_format_falls_back_but_truncation_does_not(monkeypatch):
    client = AdbClient("fake-adb", "test-device")
    png = cv2.imencode(".png", np.zeros((6, 8, 3), np.uint8))[1].tobytes()
    run = Mock(side_effect=[struct.pack("<III", 8, 6, 4), png, png])
    monkeypatch.setattr(client, "_run", run)
    assert client.capture().transport == "png"
    assert client.capture().width == 8
    assert run.call_args_list[0].args[0] == ["exec-out", "screencap"]
    client2 = AdbClient("fake-adb", "test-device")
    monkeypatch.setattr(client2, "_run", Mock(return_value=struct.pack("<III", 8, 6, 1) + b"broken"))
    with pytest.raises(AdbError, match="截断"):
        client2.capture()
    assert client2._run.call_count == 1
    with pytest.raises(RawFormatError):
        decode_raw(struct.pack("<III", 4, 4, 99))


@pytest.mark.parametrize(
    "error,message",
    [
        (b"error: device unauthorized", "尚未授权"),
        (b"error: device offline", "离线"),
        (b"java.lang.SecurityException: INJECT_EVENTS", "USB 调试（安全设置）"),
        (b"error: device 'test-device' not found", "操作失败"),
    ],
)
def test_adb_error_is_chinese_and_command_is_bound_to_device(monkeypatch, error, message):
    run = Mock(return_value=subprocess.CompletedProcess([], 1, b"", error))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AdbError, match=message.replace("（", ".").replace("）", ".")):
        AdbClient("fake-adb", "test-device").capture()
    args = run.call_args.args[0]
    assert args[:3] == ["fake-adb", "-s", "test-device"]
    assert run.call_args.kwargs["timeout"] == 8
    assert "shell" not in run.call_args.kwargs


def test_adb_timeout(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("adb", 8)))
    with pytest.raises(AdbError, match="超过 8 秒"):
        AdbClient("fake", "test-device").capture()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_input_error_preserves_original_even_with_success_exit_code(monkeypatch, frame, stream):
    raw = b"java.lang.SecurityException: requires android.permission.INJECT_EVENTS\n at InputManager"
    output = {"stdout": b"", "stderr": b""}
    output[stream] = raw
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, **output)))
    with pytest.raises(AdbError, match="INJECT_EVENTS") as caught:
        AdbClient("fake", frame.serial).tap(10, 20, frame)
    error = caught.value
    assert error.returncode == 0
    assert getattr(error, stream) == raw
    assert error.command[-5:] == ("shell", "input", "tap", "10", "20")
    assert "原始错误：java.lang.SecurityException" in str(error)


def test_other_security_exception_is_not_misreported_as_input_permission(monkeypatch):
    raw = b"java.lang.SecurityException: requires android.permission.READ_LOGS"
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 1, b"", raw)))
    with pytest.raises(AdbError) as caught:
        AdbClient("fake", "test-device")._run(["shell", "logcat", "-d"])
    assert "READ_LOGS" in str(caught.value)
    assert "USB 调试（安全设置）" not in str(caught.value)
    assert caught.value.stderr == raw


def test_devices_include_unauthorized_and_offline(monkeypatch):
    client = AdbClient()
    monkeypatch.setattr(
        client,
        "_run",
        Mock(
            return_value=b"List of devices attached\na device product:x model:Phone_X\nb unauthorized\nc offline\n"
        ),
    )
    result = client.list_devices()
    assert [(d.serial, d.state, d.model) for d in result] == [
        ("a", "device", "Phone_X"),
        ("b", "unauthorized", ""),
        ("c", "offline", ""),
    ]


@pytest.mark.parametrize("x,y", [(-1, 0), (2400, 0), (0, 1080), (True, 1), (1.2, 3)])
def test_invalid_tap_does_not_execute(frame, monkeypatch, x, y):
    client = AdbClient("fake", frame.serial)
    monkeypatch.setattr(client, "_run", Mock())
    with pytest.raises(AdbError):
        client.tap(x, y, frame)
    client._run.assert_not_called()


def test_old_or_other_device_frame_is_rejected(frame, monkeypatch):
    client = AdbClient("fake", frame.serial)
    monkeypatch.setattr(client, "_run", Mock(return_value=b""))
    for bad in (replace(frame, serial="other"), replace(frame, started=frame.started - 30)):
        with pytest.raises(AdbError):
            client.tap(1, 1, bad)
    client._run.assert_not_called()
    client.tap(2399, 1079, frame)
    assert client._run.call_args.args[0] == ["shell", "input", "tap", "2399", "1079"]
