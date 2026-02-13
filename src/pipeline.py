"""파이프라인 통합: 0단계 → 1단계 → 4단계를 연결하여 실행한다.

영상 업로드 → 구간 선택 → 프레임 추출 → 선수 세그멘테이션 → 단순 합성
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from .clip_selector import get_video_meta, make_clip_selection
from .frame_extractor import extract_frames_uniform
from .segmentation import PlayerSegmenter, apply_mask_overlay


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    n_frames: int = 12
    target_person_idx: int = 0
    yolo_model: str = "yolo11x.pt"
    sam_model: str = "sam2.1_l.pt"
    person_conf_threshold: float = 0.5
    opacity_mode: str = "uniform"  # "uniform" | "fade_in" | "fade_out"
    output_dir: str = "examples/output"


def compute_opacity(idx: int, total: int, mode: str) -> float:
    """프레임 인덱스에 따른 투명도 계산"""
    if mode == "fade_in":
        return 0.3 + 0.7 * (idx / max(1, total - 1))
    elif mode == "fade_out":
        return 1.0 - 0.7 * (idx / max(1, total - 1))
    return 1.0  # uniform


def composite_simple(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    opacity_mode: str = "uniform",
) -> np.ndarray:
    """선수 마스크를 하나의 캔버스에 합성한다 (고정 카메라 전용).

    배경은 첫 번째 프레임 기반이며, 선수를 겹쳐서 배치한다.
    """
    if not frames or not masks:
        raise ValueError("프레임과 마스크가 비어있습니다.")

    h, w = frames[0].shape[:2]
    # 배경: 첫 프레임 사용 (향후 파노라마로 대체)
    canvas = frames[0].copy().astype(np.float64)

    total = len(frames)
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        alpha = compute_opacity(i, total, opacity_mode)
        mask_f = (mask / 255.0)[:, :, np.newaxis] * alpha
        canvas = canvas * (1 - mask_f) + frame.astype(np.float64) * mask_f

    return canvas.clip(0, 255).astype(np.uint8)


def run_pipeline(
    video_path: str,
    start_sec: float,
    end_sec: float,
    config: PipelineConfig | None = None,
) -> dict:
    """전체 파이프라인 실행.

    Returns:
        {
            "meta": VideoMeta,
            "clip": ClipSelection,
            "extracted": ExtractedFrames,
            "segmentations": list[SegmentationResult],
            "composite": np.ndarray (RGB),
            "overlay_previews": list[np.ndarray],
        }
    """
    if config is None:
        config = PipelineConfig()

    # 0단계: 메타데이터 & 구간 선택
    meta = get_video_meta(video_path)
    clip = make_clip_selection(meta, start_sec, end_sec)

    # 1단계: 프레임 추출
    extracted = extract_frames_uniform(video_path, clip, meta, n_frames=config.n_frames)

    # 4단계: 세그멘테이션
    segmenter = PlayerSegmenter(
        yolo_model=config.yolo_model,
        sam_model=config.sam_model,
        person_conf_threshold=config.person_conf_threshold,
    )
    seg_results = segmenter.segment_frames(extracted.frames, config.target_person_idx)

    # 오버레이 프리뷰
    overlay_previews = [
        apply_mask_overlay(sr.frame, sr.mask)
        for sr in seg_results
    ]

    # 단순 합성
    composite = composite_simple(
        [sr.frame for sr in seg_results],
        [sr.mask for sr in seg_results],
        opacity_mode=config.opacity_mode,
    )

    # 결과 저장
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(out_dir / "composite.png"),
        cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
    )

    return {
        "meta": meta,
        "clip": clip,
        "extracted": extracted,
        "segmentations": seg_results,
        "composite": composite,
        "overlay_previews": overlay_previews,
    }
