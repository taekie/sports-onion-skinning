"""1단계: 프레임 추출 (Video Frame Extraction)

선택된 구간에서 일정 간격 또는 지정 개수로 프레임을 추출한다.
고FPS 영상에서 불필요한 프레임을 줄이기 위해 간격 조절을 지원한다.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from .clip_selector import VideoMeta, ClipSelection


@dataclass
class ExtractedFrames:
    """추출된 프레임 결과"""
    frames: list[np.ndarray]  # RGB 프레임 리스트
    frame_indices: list[int]  # 원본 영상 기준 프레임 인덱스
    timestamps: list[float]   # 각 프레임의 타임스탬프 (초)


def extract_frames_uniform(
    video_path: str,
    clip: ClipSelection,
    meta: VideoMeta,
    n_frames: int = 15,
) -> ExtractedFrames:
    """선택 구간에서 균등 간격으로 n_frames개 프레임을 추출한다.

    Args:
        video_path: 영상 파일 경로
        clip: 선택된 구간 정보
        meta: 영상 메타데이터
        n_frames: 추출할 프레임 수 (기본 15)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    # 균등 간격 인덱스 계산
    if n_frames <= 1:
        indices = [clip.start_frame]
    else:
        step = (clip.end_frame - clip.start_frame) / (n_frames - 1)
        indices = [int(clip.start_frame + i * step) for i in range(n_frames)]

    frames = []
    timestamps = []
    actual_indices = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            actual_indices.append(idx)
            timestamps.append(idx / meta.fps)

    cap.release()

    return ExtractedFrames(
        frames=frames,
        frame_indices=actual_indices,
        timestamps=timestamps,
    )


def extract_frames_by_interval(
    video_path: str,
    clip: ClipSelection,
    meta: VideoMeta,
    interval_sec: float = 0.2,
) -> ExtractedFrames:
    """선택 구간에서 시간 간격(초) 기준으로 프레임을 추출한다.

    Args:
        video_path: 영상 파일 경로
        clip: 선택된 구간 정보
        meta: 영상 메타데이터
        interval_sec: 프레임 추출 간격 (초)
    """
    n_frames = max(1, int(clip.duration_sec / interval_sec) + 1)
    return extract_frames_uniform(video_path, clip, meta, n_frames)


def extract_all_frames(
    video_path: str,
    clip: ClipSelection,
    meta: VideoMeta,
) -> ExtractedFrames:
    """선택 구간의 모든 프레임을 추출한다. 주의: 프레임 수가 많을 수 있음."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)

    frames = []
    indices = []
    timestamps = []

    for idx in range(clip.start_frame, clip.end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        indices.append(idx)
        timestamps.append(idx / meta.fps)

    cap.release()

    return ExtractedFrames(
        frames=frames,
        frame_indices=indices,
        timestamps=timestamps,
    )


def save_frames(extracted: ExtractedFrames, output_dir: str, prefix: str = "frame") -> list[str]:
    """추출된 프레임들을 파일로 저장한다."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, frame in enumerate(extracted.frames):
        path = out / f"{prefix}_{i:04d}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        saved_paths.append(str(path))

    return saved_paths
