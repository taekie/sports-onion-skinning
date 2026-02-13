"""6단계: 어노테이션 (Annotation & Labels)

합성된 이미지에 높이 눈금, 트릭 이름, 궤적선,
프레임 번호 등의 정보를 추가한다.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass


@dataclass
class AnnotationConfig:
    """어노테이션 설정"""
    show_frame_numbers: bool = True
    show_trajectory: bool = True
    show_height_guide: bool = False
    show_labels: bool = False
    trajectory_color: tuple[int, int, int] = (255, 200, 0)
    trajectory_thickness: int = 2
    frame_number_color: tuple[int, int, int] = (255, 255, 255)
    frame_number_bg_color: tuple[int, int, int] = (0, 0, 0)
    frame_number_size: float = 0.5
    height_guide_color: tuple[int, int, int] = (200, 200, 200)
    label_text: str = ""
    label_position: str = "top-left"  # top-left, top-right, bottom-left, bottom-right


def annotate_frame_numbers(
    image: np.ndarray,
    centers: list[tuple[int, int] | None],
    timestamps: list[float] | None = None,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.5,
) -> np.ndarray:
    """각 선수 위치에 프레임 번호를 표시한다.

    Args:
        image: 합성 이미지 (RGB)
        centers: 각 프레임의 선수 중심 좌표
        timestamps: 각 프레임의 타임스탬프 (초). None이면 인덱스만 표시.
        color: 텍스트 색상
        bg_color: 배경 색상
        font_scale: 폰트 크기
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, center in enumerate(centers):
        if center is None:
            continue

        cx, cy = center
        if timestamps is not None and i < len(timestamps):
            label = f"#{i} ({timestamps[i]:.2f}s)"
        else:
            label = f"#{i}"

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # 텍스트 위치: 선수 중심 위쪽
        tx = cx - tw // 2
        ty = cy - 20

        # 경계 체크
        tx = max(0, min(tx, result.shape[1] - tw))
        ty = max(th + 2, min(ty, result.shape[0] - 2))

        # 배경 박스
        cv2.rectangle(
            result,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + baseline + 2),
            bg_color, -1,
        )
        # 텍스트
        cv2.putText(result, label, (tx, ty), font, font_scale, color, 1, cv2.LINE_AA)

    return result


def annotate_trajectory(
    image: np.ndarray,
    centers: list[tuple[int, int] | None],
    color: tuple[int, int, int] = (255, 200, 0),
    thickness: int = 2,
    style: str = "curve",
) -> np.ndarray:
    """선수의 이동 궤적을 그린다.

    Args:
        image: 합성 이미지
        centers: 선수 중심 좌표 리스트
        color: 궤적 색상
        thickness: 선 두께
        style: 'line' (직선), 'curve' (부드러운 곡선), 'dashed' (점선)
    """
    result = image.copy()
    valid = [(i, c) for i, c in enumerate(centers) if c is not None]

    if len(valid) < 2:
        return result

    points = np.array([c for _, c in valid], dtype=np.int32)

    if style == "curve" and len(points) >= 3:
        # 부드러운 곡선 (B-spline 근사)
        # 포인트가 충분하면 곡선 피팅
        t = np.linspace(0, 1, len(points))
        t_smooth = np.linspace(0, 1, len(points) * 10)

        # 선형 보간으로 부드러운 점 생성
        x_smooth = np.interp(t_smooth, t, points[:, 0])
        y_smooth = np.interp(t_smooth, t, points[:, 1])
        smooth_pts = np.column_stack([x_smooth, y_smooth]).astype(np.int32)
        smooth_pts = smooth_pts.reshape(-1, 1, 2)
        cv2.polylines(result, [smooth_pts], False, color, thickness, cv2.LINE_AA)
    elif style == "dashed":
        for i in range(len(points) - 1):
            p1 = tuple(points[i])
            p2 = tuple(points[i + 1])
            _draw_dashed_line(result, p1, p2, color, thickness)
    else:
        cv2.polylines(result, [points], False, color, thickness, cv2.LINE_AA)

    # 시작/끝 마커
    if valid:
        cv2.circle(result, valid[0][1], 6, color, -1, cv2.LINE_AA)
        cv2.circle(result, valid[-1][1], 8, color, 2, cv2.LINE_AA)

    return result


def _draw_dashed_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 10,
    gap_length: int = 5,
):
    """점선을 그린다."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = np.sqrt(dx**2 + dy**2)

    if dist == 0:
        return

    n_segments = int(dist / (dash_length + gap_length))
    for i in range(n_segments + 1):
        t_start = i * (dash_length + gap_length) / dist
        t_end = min(1.0, (i * (dash_length + gap_length) + dash_length) / dist)

        start = (int(pt1[0] + dx * t_start), int(pt1[1] + dy * t_start))
        end = (int(pt1[0] + dx * t_end), int(pt1[1] + dy * t_end))
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def annotate_height_guide(
    image: np.ndarray,
    centers: list[tuple[int, int] | None],
    color: tuple[int, int, int] = (200, 200, 200),
    n_lines: int = 5,
) -> np.ndarray:
    """높이 기준 가이드라인을 그린다.

    선수 궤적의 최고/최저점을 기준으로 높이 눈금을 표시한다.
    """
    result = image.copy()
    valid_centers = [c for c in centers if c is not None]

    if len(valid_centers) < 2:
        return result

    ys = [c[1] for c in valid_centers]
    xs = [c[0] for c in valid_centers]
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)

    height_range = y_max - y_min
    if height_range < 10:
        return result

    # 높이 가이드라인
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n_lines + 1):
        y = int(y_min + i * height_range / n_lines)
        # 점선 형태로 가이드라인
        for x in range(max(0, x_min - 30), min(result.shape[1], x_max + 30), 15):
            x_end = min(x + 8, result.shape[1])
            cv2.line(result, (x, y), (x_end, y), color, 1, cv2.LINE_AA)

        # 상대 높이 라벨 (위가 높은 값)
        relative_height = 1.0 - (i / n_lines)
        label = f"{relative_height * 100:.0f}%"
        cv2.putText(result, label, (max(0, x_min - 60), y + 4),
                    font, 0.35, color, 1, cv2.LINE_AA)

    return result


def annotate_label(
    image: np.ndarray,
    text: str,
    position: str = "top-left",
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.8,
    padding: int = 10,
) -> np.ndarray:
    """이미지에 텍스트 라벨을 추가한다.

    Args:
        image: 합성 이미지
        text: 라벨 텍스트 (트릭 이름 등)
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        color: 텍스트 색상
        bg_color: 배경 색상
        font_scale: 폰트 크기
        padding: 여백
    """
    if not text.strip():
        return image

    result = image.copy()
    h, w = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = text.split("\n")
    line_sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in lines]
    max_tw = max(s[0] for s in line_sizes)
    line_height = max(s[1] for s in line_sizes) + 8
    total_th = line_height * len(lines)

    # 위치 계산
    if position == "top-left":
        x0, y0 = padding, padding
    elif position == "top-right":
        x0, y0 = w - max_tw - padding * 3, padding
    elif position == "bottom-left":
        x0, y0 = padding, h - total_th - padding * 3
    else:  # bottom-right
        x0, y0 = w - max_tw - padding * 3, h - total_th - padding * 3

    # 반투명 배경
    overlay = result.copy()
    cv2.rectangle(
        overlay,
        (x0 - padding, y0 - padding),
        (x0 + max_tw + padding, y0 + total_th + padding),
        bg_color, -1,
    )
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

    # 텍스트
    for i, line in enumerate(lines):
        ty = y0 + (i + 1) * line_height
        cv2.putText(result, line, (x0, ty), font, font_scale, color, 1, cv2.LINE_AA)

    return result


def annotate_image(
    image: np.ndarray,
    centers: list[tuple[int, int] | None],
    config: AnnotationConfig | None = None,
    timestamps: list[float] | None = None,
) -> np.ndarray:
    """설정에 따라 이미지에 모든 어노테이션을 적용한다.

    Args:
        image: 합성 이미지 (RGB)
        centers: 각 프레임의 선수 중심 좌표
        config: 어노테이션 설정 (None이면 기본값)
        timestamps: 각 프레임의 타임스탬프

    Returns:
        어노테이션이 추가된 이미지
    """
    if config is None:
        config = AnnotationConfig()

    result = image.copy()

    if config.show_height_guide:
        result = annotate_height_guide(
            result, centers,
            color=config.height_guide_color,
        )

    if config.show_trajectory:
        result = annotate_trajectory(
            result, centers,
            color=config.trajectory_color,
            thickness=config.trajectory_thickness,
        )

    if config.show_frame_numbers:
        result = annotate_frame_numbers(
            result, centers,
            timestamps=timestamps,
            color=config.frame_number_color,
            bg_color=config.frame_number_bg_color,
            font_scale=config.frame_number_size,
        )

    if config.show_labels and config.label_text:
        result = annotate_label(
            result, config.label_text,
            position=config.label_position,
        )

    return result
