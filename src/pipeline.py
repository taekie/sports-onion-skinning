"""파이프라인 통합: 전체 단계를 연결하여 실행한다.

모드 1 (고정 카메라): 0→1→4→단순합성
모드 2 (카메라 모션): 0→1→4→2→3→5→6 (호모그래피+파노라마+고급합성+어노테이션)
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from .clip_selector import get_video_meta, make_clip_selection
from .frame_extractor import extract_frames_uniform
from .segmentation import PlayerSegmenter, apply_mask_overlay
from .homography import compute_cumulative_homographies, compute_canvas_size
from .panorama import build_panorama
from .compositor import composite_on_panorama, draw_trajectory, CompositeResult
from .annotator import annotate_image, AnnotationConfig


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    n_frames: int = 12
    target_person_idx: int = 0
    yolo_model: str = "yolo11x.pt"
    sam_model: str = "sam2.1_l.pt"
    person_conf_threshold: float = 0.5
    opacity_mode: str = "uniform"  # "uniform" | "fade_in" | "fade_out" | "center_focus"
    output_dir: str = "examples/output"
    # 카메라 모션 관련
    use_homography: bool = False
    feature_method: str = "sift"  # "sift" | "orb"
    panorama_method: str = "average"  # "average" | "median"
    edge_feather: int = 3
    # 어노테이션 관련
    show_frame_numbers: bool = False
    show_trajectory: bool = False
    show_height_guide: bool = False
    label_text: str = ""


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
            "panorama": np.ndarray | None,
            "homography_result": HomographyResult | None,
            "annotated": np.ndarray | None,
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

    frames = [sr.frame for sr in seg_results]
    masks = [sr.mask for sr in seg_results]

    panorama_img = None
    homography_result = None
    composite_result = None
    annotated = None

    if config.use_homography:
        # 2단계: 호모그래피 추정 (선수 마스크 제외)
        homography_result = compute_cumulative_homographies(
            frames, masks,
            method=config.feature_method,
        )

        # 캔버스 크기 계산
        canvas_w, canvas_h, offset = compute_canvas_size(
            frames[0].shape, homography_result.cumulative
        )
        canvas_size = (canvas_w, canvas_h)

        # 3단계: 배경 파노라마 생성
        pano_result = build_panorama(
            frames, masks, homography_result.cumulative,
            offset, canvas_size,
            method=config.panorama_method,
        )
        panorama_img = pano_result.panorama

        # 5단계: 파노라마 위에 선수 합성
        composite_result = composite_on_panorama(
            panorama_img, frames, masks,
            homography_result.cumulative,
            offset, canvas_size,
            opacity_mode=config.opacity_mode,
            edge_feather=config.edge_feather,
        )
        composite = composite_result.image

        # 6단계: 어노테이션
        ann_config = AnnotationConfig(
            show_frame_numbers=config.show_frame_numbers,
            show_trajectory=config.show_trajectory,
            show_height_guide=config.show_height_guide,
            label_text=config.label_text,
            show_labels=bool(config.label_text),
        )
        annotated = annotate_image(
            composite, composite_result.player_centers,
            config=ann_config,
            timestamps=extracted.timestamps,
        )
    else:
        # 단순 합성 (고정 카메라)
        composite = composite_simple(frames, masks, opacity_mode=config.opacity_mode)

    # 결과 저장
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_image = annotated if annotated is not None else composite
    cv2.imwrite(
        str(out_dir / "composite.png"),
        cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR),
    )

    if panorama_img is not None:
        cv2.imwrite(
            str(out_dir / "panorama.png"),
            cv2.cvtColor(panorama_img, cv2.COLOR_RGB2BGR),
        )

    return {
        "meta": meta,
        "clip": clip,
        "extracted": extracted,
        "segmentations": seg_results,
        "composite": composite,
        "overlay_previews": overlay_previews,
        "panorama": panorama_img,
        "homography_result": homography_result,
        "annotated": annotated,
    }
