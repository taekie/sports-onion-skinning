"""2단계: 카메라 모션 추정 (Homography Estimation)

프레임 간 특징점 매칭을 통해 변환 행렬을 추정한다.
선수 마스크를 제외한 배경 영역만 사용하여 매칭 정확도를 높인다.

변환 모드:
- similarity: 이동+회전+균일스케일 (4 DOF) — 팬+줌에 최적
- affine: 이동+회전+스케일+전단 (6 DOF)
- homography: 완전 사영 변환 (8 DOF) — 고정 카메라/단순 팬용
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class HomographyResult:
    """프레임 간 변환 추정 결과"""
    homographies: list[np.ndarray]       # 각 프레임 → 기준 프레임 변환 행렬 (3x3)
    cumulative: list[np.ndarray]         # 누적 변환 (기준 프레임 좌표계)
    ref_index: int                       # 기준 프레임 인덱스
    inlier_counts: list[int]             # 각 매칭에서의 인라이어 수
    match_counts: list[int]              # 각 매칭에서의 총 매칭 수
    transform_type: str = "similarity"   # 사용된 변환 타입


def detect_and_compute(
    frame: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "sift",
    max_features: int = 5000,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """프레임에서 특징점을 검출하고 디스크립터를 계산한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    bg_mask = None
    if mask is not None:
        bg_mask = cv2.bitwise_not(mask)

    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
    else:
        detector = cv2.ORB_create(nfeatures=max_features)

    keypoints, descriptors = detector.detectAndCompute(gray, bg_mask)
    return keypoints, descriptors


def match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    method: str = "sift",
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """두 프레임의 디스크립터를 매칭한다 (Lowe's ratio test 적용)."""
    if des1 is None or des2 is None:
        return []
    if len(des1) < 2 or len(des2) < 2:
        return []

    if method == "sift":
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

    return good


def _affine_to_3x3(M: np.ndarray) -> np.ndarray:
    """2x3 아핀/유사 변환 행렬을 3x3 호모그래피 형식으로 변환한다."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def estimate_transform(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    transform_type: str = "similarity",
    ransac_threshold: float = 5.0,
    min_matches: int = 10,
) -> tuple[np.ndarray | None, int]:
    """매칭된 특징점으로부터 변환 행렬을 추정한다.

    Args:
        transform_type: 'similarity' (4DOF), 'affine' (6DOF), 'homography' (8DOF)
        ransac_threshold: RANSAC 오차 임계값 (픽셀)
        min_matches: 최소 매칭 수

    Returns:
        (변환 행렬 3x3, 인라이어 수) 또는 (None, 0)
    """
    if len(matches) < min_matches:
        return None, 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if transform_type == "similarity":
        # 4 DOF: 이동 + 회전 + 균일 스케일 — 팬+줌에 최적
        M, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )
        if M is None:
            return None, 0
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        return _affine_to_3x3(M), inliers

    elif transform_type == "affine":
        # 6 DOF: 이동 + 회전 + 비균일 스케일 + 전단
        M, inlier_mask = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )
        if M is None:
            return None, 0
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        return _affine_to_3x3(M), inliers

    else:
        # 8 DOF: 완전 사영 변환
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        inliers = int(mask.sum()) if mask is not None else 0
        return H, inliers


def compute_pairwise_homographies(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    method: str = "sift",
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 5.0,
    transform_type: str = "similarity",
) -> list[tuple[np.ndarray | None, int, int]]:
    """인접 프레임 간의 변환 행렬을 순차적으로 계산한다.

    Returns:
        [(H_i→i+1, inlier_count, match_count), ...]
    """
    results = []

    features = []
    for i, frame in enumerate(frames):
        m = masks[i] if masks is not None else None
        kp, des = detect_and_compute(frame, m, method=method)
        features.append((kp, des))

    for i in range(len(frames) - 1):
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]

        matches = match_features(des1, des2, method=method, ratio_threshold=ratio_threshold)
        H, inliers = estimate_transform(
            kp1, kp2, matches,
            transform_type=transform_type,
            ransac_threshold=ransac_threshold,
        )

        results.append((H, inliers, len(matches)))

    return results


def _invert_3x3(H: np.ndarray) -> np.ndarray:
    """3x3 변환 행렬의 역행렬을 구한다. 아핀/유사 변환도 안전하게 처리."""
    # 마지막 행이 [0, 0, 1]인 아핀 변환의 경우도 안전
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)


def _extract_scale(H: np.ndarray) -> float:
    """3x3 변환 행렬에서 스케일 팩터를 추출한다."""
    # SVD로 스케일 추출 (2x2 부분)
    M = H[:2, :2]
    sx = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    sy = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
    return (sx + sy) / 2.0


def normalize_scales(
    cumulative: list[np.ndarray],
    ref_index: int,
    target_scale: float = 1.0,
) -> list[np.ndarray]:
    """누적 변환의 스케일을 정규화한다.

    줌에 의한 크기 차이를 제거하여 모든 프레임이 동일한 스케일로 워핑되도록 한다.

    Args:
        cumulative: 누적 변환 행렬 리스트
        ref_index: 기준 프레임 인덱스
        target_scale: 목표 스케일 (1.0 = 기준 프레임 크기 유지)
    """
    normalized = []
    for i, H in enumerate(cumulative):
        if H is None:
            normalized.append(np.eye(3, dtype=np.float64))
            continue

        scale = _extract_scale(H)
        if scale < 0.01:
            normalized.append(H.copy())
            continue

        # 스케일 보정: target_scale / current_scale
        correction = target_scale / scale
        S = np.eye(3, dtype=np.float64)
        S[0, 0] = correction
        S[1, 1] = correction
        normalized.append(S @ H)

    return normalized


def compute_cumulative_homographies(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    ref_index: int | None = None,
    method: str = "sift",
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 5.0,
    transform_type: str = "similarity",
    normalize_zoom: bool = True,
) -> HomographyResult:
    """모든 프레임을 기준 프레임 좌표계로 변환하는 누적 변환을 계산한다.

    Args:
        frames: RGB 프레임 리스트
        masks: 선수 마스크 리스트 (선택, 255=선수)
        ref_index: 기준 프레임 인덱스 (None이면 중간 프레임)
        method: 특징점 검출 방법 ('sift' 또는 'orb')
        ratio_threshold: Lowe's ratio test 임계값
        ransac_threshold: RANSAC 임계값
        transform_type: 'similarity' (팬+줌), 'affine', 'homography'
        normalize_zoom: True면 줌 스케일을 정규화하여 일관된 크기 유지
    """
    n = len(frames)
    if n == 0:
        return HomographyResult([], [], 0, [], [])

    if ref_index is None:
        ref_index = n // 2

    pairwise = compute_pairwise_homographies(
        frames, masks, method, ratio_threshold, ransac_threshold, transform_type
    )

    cumulative = [None] * n
    cumulative[ref_index] = np.eye(3, dtype=np.float64)

    homographies = [None] * n
    homographies[ref_index] = np.eye(3, dtype=np.float64)

    inlier_counts = [0] * n
    match_counts = [0] * n
    inlier_counts[ref_index] = -1
    match_counts[ref_index] = -1

    # 기준 프레임에서 왼쪽으로
    for i in range(ref_index - 1, -1, -1):
        H_i_to_next, inliers, matches = pairwise[i]
        inlier_counts[i] = inliers
        match_counts[i] = matches

        if H_i_to_next is not None:
            homographies[i] = H_i_to_next
            cumulative[i] = cumulative[i + 1] @ H_i_to_next
        else:
            homographies[i] = np.eye(3, dtype=np.float64)
            cumulative[i] = cumulative[i + 1].copy()

    # 기준 프레임에서 오른쪽으로
    for i in range(ref_index + 1, n):
        H_prev_to_i, inliers, matches = pairwise[i - 1]
        inlier_counts[i] = inliers
        match_counts[i] = matches

        if H_prev_to_i is not None:
            homographies[i] = H_prev_to_i
            H_i_to_prev = _invert_3x3(H_prev_to_i)
            cumulative[i] = cumulative[i - 1] @ H_i_to_prev
        else:
            homographies[i] = np.eye(3, dtype=np.float64)
            cumulative[i] = cumulative[i - 1].copy()

    # 줌 정규화
    if normalize_zoom:
        cumulative = normalize_scales(cumulative, ref_index)

    return HomographyResult(
        homographies=homographies,
        cumulative=cumulative,
        ref_index=ref_index,
        inlier_counts=inlier_counts,
        match_counts=match_counts,
        transform_type=transform_type,
    )


def compute_canvas_size(
    frame_shape: tuple[int, int],
    cumulative_homographies: list[np.ndarray],
) -> tuple[int, int, np.ndarray]:
    """모든 프레임을 워핑했을 때 필요한 캔버스 크기와 오프셋을 계산한다.

    Returns:
        (canvas_width, canvas_height, offset_matrix)
    """
    h, w = frame_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    all_corners = []
    for H in cumulative_homographies:
        if H is not None:
            warped = cv2.perspectiveTransform(corners, H)
            all_corners.append(warped)

    if not all_corners:
        return w, h, np.eye(3, dtype=np.float64)

    all_pts = np.concatenate(all_corners, axis=0)
    x_min, y_min = all_pts.min(axis=0).ravel()
    x_max, y_max = all_pts.max(axis=0).ravel()

    offset = np.eye(3, dtype=np.float64)
    offset[0, 2] = -x_min
    offset[1, 2] = -y_min

    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))

    return canvas_w, canvas_h, offset
