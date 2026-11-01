"""验证实时短提示采样、并行顺序、有界积压以及停止后的尾部处理。"""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from doudizhu_lab.runtime.analysis import MediaSampler, OrderedAnalyzer
from doudizhu_lab.vision.cards import ink_mask, white_mask


def test_faster_channel_masks_preserve_exact_threshold_pixels():
    rgb = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    for offset, value in enumerate((89, 90, 100, 144, 145, 155, 156, 164, 165, 255)):
        rgb[offset] = value
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    old_white = ((minimum > 155) & (maximum - minimum < 65)).astype(np.uint8) * 255
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    old_ink = (
        (np.maximum.reduce([r, g, b]) < 145) | ((r * 10 > g * 14) & (r * 10 > b * 13) & (g < 165) & (r > 100))
    ).astype(np.uint8) * 255
    np.testing.assert_array_equal(white_mask(rgb), old_white)
    np.testing.assert_array_equal(ink_mask(rgb), old_ink)


def test_media_sampling_keeps_ten_fps_on_fifteen_fps_input(frame):
    sampler = MediaSampler()
    selected = []
    for i in range(150):
        current = replace(frame, stream_id="s", media_pts_us=round(i * 1_000_000 / 15))
        if sampler.accept(current):
            selected.append(current.media_pts_us)
        assert not sampler.accept(current)
    assert len(selected) == 100
    assert max(b - a for a, b in zip(selected, selected[1:])) < 134_000
    assert sampler.accept(replace(frame, stream_id="new", media_pts_us=0))
    assert not sampler.accept(replace(frame, stream_id="new", media_pts_us=-1))


def test_continuous_sampling_spans_stability_window_without_rounding_short_pairs(frame):
    sampler = MediaSampler(minimum_spacing=True)
    selected = []
    for i in range(150):
        current = replace(frame, stream_id="s", media_pts_us=round(i * 1_000_000 / 15))
        if sampler.accept(current):
            selected.append(current.media_pts_us)
    assert 70 <= len(selected) <= 80
    assert all(b - a >= 100_000 for a, b in zip(selected, selected[1:]))
    # 本次真机短提示只有两帧被旧采样命中，相隔94980us，不能改成100ms通过。
    sampler = MediaSampler(minimum_spacing=True)
    accepted = [
        t
        for t in (0, 94_980, 133_000, 200_000, 266_000)
        if sampler.accept(replace(frame, stream_id="real-jitter", media_pts_us=t))
    ]
    assert accepted == [0, 133_000, 266_000]


def test_parallel_completion_is_ordered_and_backlog_is_bounded(frame):
    release_first, completed_second = Event(), Event()

    def factory():
        def recognize(value):
            if value.frame_id == "first":
                assert release_first.wait(3)
            else:
                completed_second.set()
            return value.frame_id

        return recognize

    pool = OrderedAnalyzer(factory, capacity=2)
    try:
        assert pool.submit(replace(frame, frame_id="first"))
        assert pool.submit(replace(frame, frame_id="second"))
        assert completed_second.wait(3)
        assert pool.size == 2
        assert not pool.submit(replace(frame, frame_id="third"))
        pool.close()
        release_first.set()
        assert [pool.take()[1], pool.take()[1]] == ["first", "second"]
        assert pool.take() is None and pool.size == 0
        assert not pool.submit(frame)
    finally:
        release_first.set()
        pool.shutdown()


def test_worker_failure_is_reported_instead_of_success(frame):
    def factory():
        def fail(_):
            raise ValueError("模拟识别故障")

        return fail

    pool = OrderedAnalyzer(factory)
    try:
        assert pool.submit(frame)
        with pytest.raises(ValueError, match="模拟识别故障"):
            pool.take()
    finally:
        pool.shutdown()


def test_idle_consumer_can_be_stopped_without_a_frame():
    pool = OrderedAnalyzer(lambda: lambda value: value)
    pool.close()
    assert pool.take() is None
    pool.shutdown()
