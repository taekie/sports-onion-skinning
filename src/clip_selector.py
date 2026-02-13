"""0단계: 영상 업로드 & 구간 선택 (Video Upload & Clip Selection)

영상 파일을 로드하고, 어니언스키닝할 구간(시작/끝 시점)을 선택하여
해당 구간의 메타데이터와 트리밍된 프레임들을 반환한다.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMeta:
    """영상 메타데이터"""
    path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    codec: str


@dataclass
class ClipSelection:
    """선택된 구간 정보"""
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    duration_sec: float
    estimated_frames: int


def get_video_meta(video_path: str) -> VideoMeta:
    """영상 메타데이터를 읽어온다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    duration_sec = total_frames / fps if fps > 0 else 0

    cap.release()

    return VideoMeta(
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        width=width,
        height=height,
        codec=codec,
    )


def make_clip_selection(meta: VideoMeta, start_sec: float, end_sec: float) -> ClipSelection:
    """시작/끝 시점으로부터 ClipSelection을 생성한다."""
    start_sec = max(0.0, start_sec)
    end_sec = min(meta.duration_sec, end_sec)
    if end_sec <= start_sec:
        raise ValueError(f"끝 시점({end_sec}s)이 시작 시점({start_sec}s)보다 같거나 앞섭니다.")

    start_frame = int(start_sec * meta.fps)
    end_frame = int(end_sec * meta.fps)
    duration = end_sec - start_sec
    estimated_frames = end_frame - start_frame

    return ClipSelection(
        start_sec=start_sec,
        end_sec=end_sec,
        start_frame=start_frame,
        end_frame=end_frame,
        duration_sec=duration,
        estimated_frames=estimated_frames,
    )


def generate_thumbnail_strip(video_path: str, n_thumbnails: int = 8,
                              thumb_width: int = 160) -> list[np.ndarray]:
    """타임라인 썸네일 스트립 생성. 균등 간격으로 n_thumbnails개를 추출한다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = [int(i * total / n_thumbnails) for i in range(n_thumbnails)]
    thumbnails = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            thumb_height = int(thumb_width * h / w)
            thumb = cv2.resize(frame, (thumb_width, thumb_height))
            # BGR → RGB
            thumbnails.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

    cap.release()
    return thumbnails


def generate_preview_at(video_path: str, time_sec: float) -> np.ndarray | None:
    """특정 시점의 프리뷰 프레임을 반환한다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None
