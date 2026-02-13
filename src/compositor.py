"""5단계: 합성 & 블렌딩 (Compositing)

파노라마 배경 위에 선수 마스크를 호모그래피 좌표계로 배치하고,
다양한 블렌딩 효과를 적용한다.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from .panorama import warp_frame, warp_mask


@dataclass
class CompositeResult:
    """합성 결과"""
    image: np.ndarray                    # 최종 합성 이미지 (RGB)
    panorama_bg: np.ndarray              # 사용된 배경 파노라마
    player_centers: list[tuple[int, int]]  # 각 선수의 중심 좌표 (파노라마 좌표계)


def compute_opacity(idx: int, total: int, mode: str) -> float:
    """프레임 인덱스에 따른 투명도 계산.

    Args:
        idx: 현재 프레임 인덱스
        total: 전체 프레임 수
        mode: 'uniform', 'fade_in', 'fade_out', 'center_focus'
    """
    if total <= 1:
        return 1.0

    t = idx / (total - 1)

    if mode == "fade_in":
        return 0.3 + 0.7 * t
    elif mode == "fade_out":
        return 1.0 - 0.7 * t
    elif mode == "center_focus":
        # 중간 프레임이 가장 불투명, 양 끝은 반투명
        return 0.4 + 0.6 * (1.0 - abs(2.0 * t - 1.0))
    return 1.0  # uniform


def get_mask_center(mask: np.ndarray) -> tuple[int, int] | None:
    """마스크의 무게중심(center of mass)을 계산한다."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def composite_on_panorama(
    panorama: np.ndarray,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    homographies: list[np.ndarray],
    offset: np.ndarray,
    canvas_size: tuple[int, int],
    opacity_mode: str = "uniform",
    edge_feather: int = 3,
) -> CompositeResult:
    """파노라마 배경 위에 선수를 합성한다.

    Args:
        panorama: 배경 파노라마 (RGB)
        frames: 원본 프레임 리스트
        masks: 선수 마스크 리스트 (255=선수)
        homographies: 누적 호모그래피 리스트
        offset: 캔버스 오프셋 행렬
        canvas_size: (width, height)
        opacity_mode: 투명도 모드
        edge_feather: 마스크 경계 페더링 크기 (부드러운 경계)

    Returns:
        CompositeResult
    """
    canvas = panorama.copy().astype(np.float64)
    total = len(frames)
    player_centers = []

    for i in range(total):
        # 선수 프레임과 마스크를 파노라마 좌표계로 워핑
        warped_player = warp_frame(frames[i], homographies[i], offset, canvas_size)
        warped_mask = warp_mask(masks[i], homographies[i], offset, canvas_size)

        # 마스크 경계 페더링 (부드러운 합성)
        if edge_feather > 0:
            warped_mask_f = warped_mask.astype(np.float64) / 255.0
            warped_mask_f = cv2.GaussianBlur(
                warped_mask_f, (edge_feather * 2 + 1, edge_feather * 2 + 1), 0
            )
        else:
            warped_mask_f = warped_mask.astype(np.float64) / 255.0

        # 투명도 적용
        alpha = compute_opacity(i, total, opacity_mode)
        mask_alpha = warped_mask_f[:, :, np.newaxis] * alpha

        # 알파 블렌딩
        canvas = canvas * (1 - mask_alpha) + warped_player.astype(np.float64) * mask_alpha

        # 중심점 기록
        center = get_mask_center(warped_mask)
        player_centers.append(center)

    result_img = canvas.clip(0, 255).astype(np.uint8)

    return CompositeResult(
        image=result_img,
        panorama_bg=panorama,
        player_centers=player_centers,
    )


def draw_trajectory(
    image: np.ndarray,
    centers: list[tuple[int, int] | None],
    color: tuple[int, int, int] = (255, 200, 0),
    thickness: int = 2,
    draw_dots: bool = True,
    dot_radius: int = 5,
) -> np.ndarray:
    """선수의 이동 궤적선을 이미지에 그린다.

    Args:
        image: 합성 이미지 (RGB)
        centers: 각 프레임의 선수 중심 좌표 리스트
        color: 궤적선 색상 (RGB)
        thickness: 선 두께
        draw_dots: 각 포인트에 점 표시 여부
        dot_radius: 점 반지름

    Returns:
        궤적이 그려진 이미지
    """
    result = image.copy()
    valid_centers = [(i, c) for i, c in enumerate(centers) if c is not None]

    if len(valid_centers) < 2:
        return result

    # 궤적선 그리기 (연속된 유효 포인트 연결)
    points = np.array([c for _, c in valid_centers], dtype=np.int32)

    # 부드러운 곡선을 위해 polylines 사용
    cv2.polylines(result, [points], isClosed=False, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)

    # 각 포인트에 점 표시
    if draw_dots:
        for i, (frame_idx, center) in enumerate(valid_centers):
            # 시간 순서에 따라 점 크기/투명도 변화
            t = i / max(1, len(valid_centers) - 1)
            r = max(2, int(dot_radius * (0.5 + 0.5 * t)))
            cv2.circle(result, center, r, color, -1, cv2.LINE_AA)

    return result


def apply_color_tint(
    image: np.ndarray,
    mask: np.ndarray,
    tint_color: tuple[int, int, int],
    intensity: float = 0.2,
) -> np.ndarray:
    """선수 영역에 색상 틴트를 적용한다."""
    result = image.copy().astype(np.float64)
    mask_f = (mask > 127)[:, :, np.newaxis].astype(np.float64)
    tint = np.array(tint_color, dtype=np.float64).reshape(1, 1, 3)

    result = result * (1 - mask_f * intensity) + tint * mask_f * intensity
    return result.clip(0, 255).astype(np.uint8)


def add_shadow(
    image: np.ndarray,
    mask: np.ndarray,
    offset: tuple[int, int] = (5, 5),
    blur_size: int = 15,
    shadow_alpha: float = 0.3,
) -> np.ndarray:
    """선수 마스크에 드롭 섀도우를 추가한다."""
    result = image.copy().astype(np.float64)
    h, w = mask.shape[:2]

    # 마스크를 오프셋만큼 이동
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shadow_mask = cv2.warpAffine(mask, M, (w, h))

    # 그림자 블러
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)
    shadow_f = (shadow_mask / 255.0)[:, :, np.newaxis] * shadow_alpha

    # 원본 마스크 영역은 그림자에서 제외
    original_f = (mask > 127)[:, :, np.newaxis].astype(np.float64)
    shadow_f = shadow_f * (1 - original_f)

    # 그림자 적용 (어둡게)
    result = result * (1 - shadow_f)
    return result.clip(0, 255).astype(np.uint8)
