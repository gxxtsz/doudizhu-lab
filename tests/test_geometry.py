"""坐标映射覆盖缩放、横竖屏、黑边和 Qt 高 DPI 逻辑坐标。"""

import pytest

from doudizhu_lab.domain.geometry import Viewport
from doudizhu_lab.domain.models import NormalizedRect


@pytest.mark.parametrize("view", [(800, 600), (1600, 900), (400, 600), (2400, 1080)])
@pytest.mark.parametrize("size", [(2400, 1080), (1080, 2400)])
def test_letterbox_mapping_and_exclusive_edges(view, size):
    vp = Viewport.fit(*view, *size)
    assert vp.to_image(vp.x + vp.width / 2, vp.y + vp.height / 2) == (size[0] // 2, size[1] // 2)
    assert vp.to_image(vp.x, vp.y) == (0, 0)
    assert vp.to_image(vp.x - 0.01, vp.y) is None
    assert vp.to_image(vp.x + vp.width, vp.y) is None
    assert vp.to_image(vp.x, vp.y + vp.height) is None


@pytest.mark.parametrize("device_pixel_ratio", [1, 1.25, 1.5, 2])
def test_physical_pixels_converted_once(device_pixel_ratio):
    vp = Viewport.fit(1000, 800, 2400, 1080)
    physical_x = (vp.x + vp.width * 0.25) * device_pixel_ratio
    physical_y = (vp.y + vp.height * 0.5) * device_pixel_ratio
    assert vp.to_image(physical_x / device_pixel_ratio, physical_y / device_pixel_ratio) == (600, 540)


def test_drag_reverse_and_bad_region():
    vp = Viewport.fit(1000, 450, 2400, 1080)
    rect = vp.selection((700, 300), (100, 100))
    assert rect is not None and rect.x == 0.1 and rect.y == pytest.approx(240 / 1080)
    assert vp.selection((-1, 10), (100, 100)) is None
    assert vp.to_image(float("nan"), 10) is None
    with pytest.raises(ValueError):
        NormalizedRect(0.5, 0.5, 0.6, 0.2)
    with pytest.raises(ValueError):
        Viewport.fit(0, 400, 2400, 1080)
