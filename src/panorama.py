"""3단계: 배경 파노라마 생성 (Background Panorama Stitching)

호모그래피로 프레임을 워핑한 후, 선수 영역을 제외하고
중앙값 기반으로 깨끗한 배경 파노라마를 생성한다.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PanoramaResult:
    """파노라마 생성 결과"""
    panorama: np.ndarray       # 배경 파노라마 이미지 (RGB)
    canvas_size: tuple[int, int]  # (width, height)
    offset_matrix: np.ndarray  # 캔버스 오프셋 변환 행렬


def warp_frame(
    frame: np.ndarray,
    homography: np.ndarray,
    offset: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """프레임을 호모그래피 + 오프셋으로 워핑한다.

    Args:
        frame: RGB 이미지
        homography: 기준 프레임 좌표계로의 변환 행렬
        offset: 캔버스 오프셋 행렬
        canvas_size: (width, height)

    Returns:
        워핑된 프레임 (canvas_size 크기)
    """
    H = offset @ homography
    warped = cv2.warpPerspective(
        frame, H, canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped


def warp_mask(
    mask: np.ndarray,
    homography: np.ndarray,
    offset: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """마스크를 호모그래피 + 오프셋으로 워핑한다.

    Args:
        mask: 바이너리 마스크 (H, W), 값 0 또는 255
        homography: 변환 행렬
        offset: 캔버스 오프셋
        canvas_size: (width, height)

    Returns:
        워핑된 마스크
    """
    H = offset @ homography
    warped = cv2.warpPerspective(
        mask, H, canvas_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def get_warp_coverage_mask(
    frame_shape: tuple[int, int],
    homography: np.ndarray,
    offset: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """워핑된 영역의 커버리지 마스크를 계산한다.

    Returns:
        커버리지 마스크 (canvas_size), 값 0 또는 255
    """
    ones = np.ones(frame_shape[:2], dtype=np.uint8) * 255
    return warp_mask(ones, homography, offset, canvas_size)


def build_panorama_median(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    homographies: list[np.ndarray],
    offset: np.ndarray,
    canvas_size: tuple[int, int],
    batch_size: int = 5,
) -> np.ndarray:
    """중앙값(median) 기반 배경 파노라마를 생성한다.

    선수 마스크 영역을 제외하고, 여러 프레임의 중앙값을 취하여
    움직이는 객체가 제거된 깨끗한 배경을 생성한다.

    Args:
        frames: RGB 프레임 리스트
        masks: 선수 마스크 리스트 (255=선수)
        homographies: 누적 호모그래피 리스트
        offset: 캔버스 오프셋 행렬
        canvas_size: (width, height)
        batch_size: 메모리 절약을 위한 배치 크기
    """
    cw, ch = canvas_size
    n = len(frames)

    # 각 픽셀 위치에서 유효한 값들을 모으기 위해 스택 사용
    # 메모리 효율을 위해 배치로 처리
    accum = np.zeros((ch, cw, 3), dtype=np.float64)
    count = np.zeros((ch, cw), dtype=np.int32)

    # 모든 워핑된 프레임과 마스크를 수집
    warped_bg_pixels = []

    for i in range(n):
        warped_frame = warp_frame(frames[i], homographies[i], offset, canvas_size)
        warped_player_mask = warp_mask(masks[i], homographies[i], offset, canvas_size)
        coverage = get_warp_coverage_mask(
            frames[i].shape, homographies[i], offset, canvas_size
        )

        # 배경 영역: 커버리지 있고 선수가 아닌 곳
        bg_mask = (coverage > 127) & (warped_player_mask < 127)

        warped_bg_pixels.append((warped_frame, bg_mask))

    # 중앙값 계산 (메모리 효율적)
    # 프레임 수가 적으면 스택으로 한번에
    if n <= 20:
        stack = np.zeros((n, ch, cw, 3), dtype=np.uint8)
        valid = np.zeros((n, ch, cw), dtype=bool)

        for i, (wf, bm) in enumerate(warped_bg_pixels):
            stack[i] = wf
            valid[i] = bm

        valid_count = valid.sum(axis=0)
        has_data = valid_count > 0

        # 유효하지 않은 픽셀을 큰 값으로 설정하여 중앙값에 영향을 줄임
        for i in range(n):
            stack[i][~valid[i]] = 0

        # 마스크 기반 중앙값 계산
        # 각 픽셀 위치에서 유효한 값들만 사용
        panorama = np.zeros((ch, cw, 3), dtype=np.uint8)

        # 빠른 근사: 가중 평균 대신 masked median
        # np.ma.median은 느리므로 sort + 인덱스 방식 사용
        sorted_stack = np.sort(stack, axis=0)
        median_idx = valid_count // 2

        for c in range(3):
            for y in range(ch):
                for x in range(cw):
                    if has_data[y, x]:
                        mid = median_idx[y, x]
                        panorama[y, x, c] = sorted_stack[mid, y, x, c]

    else:
        # 프레임이 많으면 평균 기반 (메모리 절약)
        panorama = np.zeros((ch, cw, 3), dtype=np.float64)
        count = np.zeros((ch, cw, 1), dtype=np.float64)

        for wf, bm in warped_bg_pixels:
            bm_3 = bm[:, :, np.newaxis]
            panorama += wf.astype(np.float64) * bm_3
            count += bm_3.astype(np.float64)

        count = np.maximum(count, 1)
        panorama = (panorama / count).clip(0, 255).astype(np.uint8)

    return panorama


def build_panorama_average(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    homographies: list[np.ndarray],
    offset: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """가중 평균 기반 배경 파노라마를 생성한다 (빠르고 메모리 효율적)."""
    cw, ch = canvas_size
    accum = np.zeros((ch, cw, 3), dtype=np.float64)
    count = np.zeros((ch, cw, 1), dtype=np.float64)

    for i in range(len(frames)):
        warped = warp_frame(frames[i], homographies[i], offset, canvas_size)
        warped_mask = warp_mask(masks[i], homographies[i], offset, canvas_size)
        coverage = get_warp_coverage_mask(
            frames[i].shape, homographies[i], offset, canvas_size
        )

        bg = (coverage > 127) & (warped_mask < 127)
        bg_3 = bg[:, :, np.newaxis]

        accum += warped.astype(np.float64) * bg_3
        count += bg_3.astype(np.float64)

    count = np.maximum(count, 1)
    panorama = (accum / count).clip(0, 255).astype(np.uint8)
    return panorama


def build_panorama(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    homographies: list[np.ndarray],
    offset: np.ndarray,
    canvas_size: tuple[int, int],
    method: str = "average",
) -> PanoramaResult:
    """배경 파노라마를 생성한다.

    Args:
        frames: RGB 프레임 리스트
        masks: 선수 마스크 리스트 (255=선수)
        homographies: 누적 호모그래피 리스트
        offset: 캔버스 오프셋 행렬
        canvas_size: (width, height)
        method: 'median' 또는 'average'

    Returns:
        PanoramaResult
    """
    if method == "median":
        panorama = build_panorama_median(
            frames, masks, homographies, offset, canvas_size
        )
    else:
        panorama = build_panorama_average(
            frames, masks, homographies, offset, canvas_size
        )

    return PanoramaResult(
        panorama=panorama,
        canvas_size=canvas_size,
        offset_matrix=offset,
    )
