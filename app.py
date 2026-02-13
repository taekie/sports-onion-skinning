"""스포츠 어니언스키닝 — Gradio 웹 UI

영상 업로드 → 구간 선택 → 프레임 추출 → 세그멘테이션 → 합성을
단일 인터페이스에서 수행한다.
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path

from src.clip_selector import get_video_meta, make_clip_selection, generate_thumbnail_strip
from src.frame_extractor import extract_frames_uniform
from src.segmentation import PlayerSegmenter, SegmentationResult, apply_mask_overlay
from src.pipeline import composite_simple, PipelineConfig
from src.homography import compute_cumulative_homographies, compute_canvas_size, visualize_feature_matches
from src.panorama import build_panorama
from src.compositor import composite_on_panorama, draw_trajectory
from src.annotator import annotate_image, AnnotationConfig

# ── 전역 상태 ──
_state = {
    "video_path": None,
    "meta": None,
    "extracted_frames": None,
    "seg_results": None,
    "segmenter": None,
    "homography_result": None,
    "panorama": None,
    "canvas_size": None,
    "offset": None,
    "composite_result": None,
    "correction_frame_idx": None,
    "correction_points": [],     # [(x, y, label), ...] — label: 1=선수, 0=배경
    "logo_rects": [],            # [(x1, y1, x2, y2), ...] — 로고/고정 그래픽 제외 영역
    "logo_click_start": None,    # (x, y) — 현재 그리는 사각형의 시작점
}


def on_video_upload(video_path):
    """영상 업로드 시 메타데이터를 읽고 슬라이더 범위를 설정한다."""
    if video_path is None:
        return (
            "영상을 업로드하세요.",
            gr.update(maximum=1, value=0),
            gr.update(maximum=1, value=1),
        )

    meta = get_video_meta(video_path)
    _state["video_path"] = video_path
    _state["meta"] = meta

    info = (
        f"해상도: {meta.width}×{meta.height} | "
        f"FPS: {meta.fps:.1f} | "
        f"길이: {meta.duration_sec:.1f}초 | "
        f"총 프레임: {meta.total_frames} | "
        f"코덱: {meta.codec}"
    )

    return (
        info,
        gr.update(minimum=0, maximum=meta.duration_sec, value=0, step=0.1),
        gr.update(minimum=0, maximum=meta.duration_sec, value=meta.duration_sec, step=0.1),
    )


def on_extract_frames(start_sec, end_sec, n_frames):
    """구간 선택 후 프레임을 추출하고 갤러리에 표시한다."""
    if _state["video_path"] is None or _state["meta"] is None:
        return None, [], "먼저 영상을 업로드하세요."

    meta = _state["meta"]
    clip = make_clip_selection(meta, start_sec, end_sec)
    extracted = extract_frames_uniform(_state["video_path"], clip, meta, n_frames=int(n_frames))
    _state["extracted_frames"] = extracted

    # 썸네일 스트립 생성
    thumbs = generate_thumbnail_strip(_state["video_path"], n_thumbnails=8)
    strip = np.concatenate(thumbs, axis=1) if thumbs else None

    info = (
        f"구간: {clip.start_sec:.1f}s ~ {clip.end_sec:.1f}s ({clip.duration_sec:.1f}초)\n"
        f"추출 프레임: {len(extracted.frames)}개"
    )

    gallery_images = [(frame, f"#{i} ({ts:.2f}s)") for i, (frame, ts)
                      in enumerate(zip(extracted.frames, extracted.timestamps))]

    return strip, gallery_images, info


def on_segment(yolo_model, sam_model, conf_threshold, person_idx, enhance_dark):
    """추출된 프레임들에 대해 세그멘테이션을 실행한다."""
    if _state["extracted_frames"] is None:
        return [], None, "먼저 프레임을 추출하세요."

    segmenter = PlayerSegmenter(
        yolo_model=yolo_model,
        sam_model=sam_model,
        person_conf_threshold=conf_threshold,
        enhance_dark=enhance_dark,
    )
    _state["segmenter"] = segmenter

    frames = _state["extracted_frames"].frames
    seg_results = segmenter.segment_frames(frames, target_person_idx=int(person_idx))
    _state["seg_results"] = seg_results

    # 오버레이 프리뷰
    overlays = []
    detected_count = 0
    for i, sr in enumerate(seg_results):
        overlay = apply_mask_overlay(sr.frame, sr.mask)
        label = f"#{i}"
        if sr.bbox is not None:
            label += f" (conf: {sr.confidence:.2f})"
            detected_count += 1
        else:
            label += " (미검출)"
        overlays.append((overlay, label))

    # 합성
    composite = composite_simple(
        [sr.frame for sr in seg_results],
        [sr.mask for sr in seg_results],
        opacity_mode="uniform",
    )

    info = f"세그멘테이션 완료: {detected_count}/{len(seg_results)} 프레임에서 선수 검출"

    return overlays, composite, info


def on_recomposite(opacity_mode):
    """투명도 모드를 변경하여 재합성한다."""
    if _state["seg_results"] is None:
        return None, "먼저 세그멘테이션을 실행하세요."

    composite = composite_simple(
        [sr.frame for sr in _state["seg_results"]],
        [sr.mask for sr in _state["seg_results"]],
        opacity_mode=opacity_mode,
    )
    return composite, f"합성 완료 (투명도: {opacity_mode})"


def save_result(composite_img):
    """합성 결과를 저장한다."""
    if composite_img is None:
        return "저장할 이미지가 없습니다."

    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "composite.png"

    if isinstance(composite_img, np.ndarray):
        bgr = cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    return f"저장 완료: {path}"


# ── 2탭: 수동 보정 ──

def _build_seg_gallery():
    """현재 세그멘테이션 결과로 갤러리를 재생성한다."""
    overlays = []
    detected = 0
    for i, sr in enumerate(_state["seg_results"]):
        overlay = apply_mask_overlay(sr.frame, sr.mask)
        label = f"#{i}"
        if sr.bbox is not None:
            label += f" (conf: {sr.confidence:.2f})"
            detected += 1
        else:
            label += " (미검출)"
        overlays.append((overlay, label))
    return overlays, detected


def on_select_frame_for_correction(evt: gr.SelectData):
    """갤러리에서 보정할 프레임을 선택한다."""
    if _state["seg_results"] is None:
        return None, "먼저 세그멘테이션을 실행하세요."

    idx = evt.index
    if idx >= len(_state["seg_results"]):
        return None, "잘못된 프레임 인덱스입니다."

    _state["correction_frame_idx"] = idx
    _state["correction_points"] = []

    sr = _state["seg_results"][idx]
    overlay = apply_mask_overlay(sr.frame, sr.mask)
    status = f"프레임 #{idx} 선택됨"
    if sr.bbox is None:
        status += " (미검출 — 선수 위치를 클릭하세요)"
    else:
        status += f" (conf: {sr.confidence:.2f}) — 클릭으로 보정 가능"

    return overlay, status


def on_click_correction(evt: gr.SelectData, click_mode):
    """보정 이미지를 클릭하여 포인트 프롬프트를 추가한다."""
    idx = _state["correction_frame_idx"]
    if idx is None or _state["seg_results"] is None:
        return None, "먼저 프레임을 선택하세요."
    if _state["segmenter"] is None:
        return None, "먼저 세그멘테이션을 실행하세요."

    x, y = evt.index[0], evt.index[1]
    label = 1 if click_mode == "선수 (전경)" else 0
    _state["correction_points"].append((x, y, label))

    # 누적된 포인트로 SAM 재실행
    frame = _state["seg_results"][idx].frame
    points = [[p[0], p[1]] for p in _state["correction_points"]]
    labels = [p[2] for p in _state["correction_points"]]

    new_mask = _state["segmenter"].segment_with_points(frame, points, labels)

    # 결과 업데이트
    _state["seg_results"][idx] = SegmentationResult(
        frame=frame,
        mask=new_mask,
        bbox=None,
        confidence=1.0,
    )

    overlay = apply_mask_overlay(frame, new_mask)
    # 클릭 포인트 시각화
    for px, py, pl in _state["correction_points"]:
        color = (0, 255, 0) if pl == 1 else (255, 0, 0)
        cv2.circle(overlay, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    point_info = ", ".join(
        [f"({'선수' if p[2]==1 else '배경'} {p[0]},{p[1]})" for p in _state["correction_points"]]
    )
    status = f"프레임 #{idx} — 포인트 {len(_state['correction_points'])}개: {point_info}"

    return overlay, status


def on_reset_correction():
    """현재 프레임의 보정 포인트를 초기화한다."""
    idx = _state["correction_frame_idx"]
    if idx is None or _state["seg_results"] is None:
        return None, "보정할 프레임이 없습니다."

    _state["correction_points"] = []

    # 원본 프레임으로 다시 자동 세그멘테이션
    frame = _state["seg_results"][idx].frame
    sr = _state["segmenter"].segment_frame(frame)
    _state["seg_results"][idx] = sr

    overlay = apply_mask_overlay(sr.frame, sr.mask)
    status = f"프레임 #{idx} 초기화 완료"
    return overlay, status


def on_apply_correction():
    """보정 완료 후 갤러리와 합성을 업데이트한다."""
    if _state["seg_results"] is None:
        return [], None, "세그멘테이션 결과가 없습니다."

    overlays, detected = _build_seg_gallery()

    composite = composite_simple(
        [sr.frame for sr in _state["seg_results"]],
        [sr.mask for sr in _state["seg_results"]],
        opacity_mode="uniform",
    )

    info = f"보정 적용 완료: {detected}/{len(_state['seg_results'])} 프레임에서 선수 검출"
    return overlays, composite, info


# ── 3탭: 로고 마스킹 & 호모그래피 & 파노라마 ──

def _draw_logo_rects(frame: np.ndarray) -> None:
    """프레임 위에 로고 제외 영역 사각형을 그린다 (in-place)."""
    for (x1, y1, x2, y2) in _state["logo_rects"]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 반투명 빨간색 채우기
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            red_fill = np.full_like(roi, (255, 0, 0), dtype=np.uint8)
            frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.7, red_fill, 0.3, 0)


def build_logo_mask(shape: tuple) -> np.ndarray:
    """로고 사각형들로부터 바이너리 마스크를 생성한다 (255 = 제외 영역)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in _state["logo_rects"]:
        mask[y1:y2, x1:x2] = 255
    return mask


def on_logo_preview():
    """추출된 첫 프레임을 로고 마스킹 프리뷰로 로드한다."""
    if _state["extracted_frames"] is None:
        return None, "먼저 탭 1에서 프레임을 추출하세요."

    _state["logo_rects"] = []
    _state["logo_click_start"] = None
    frame = _state["extracted_frames"].frames[0].copy()
    return frame, "프레임 로드 완료. 로고 영역의 왼쪽 위 → 오른쪽 아래를 순서대로 클릭하세요."


def on_logo_click(evt: gr.SelectData):
    """로고 영역 사각형 그리기: 첫 클릭 = 시작점, 두 번째 클릭 = 끝점."""
    if _state["extracted_frames"] is None:
        return None, "먼저 프리뷰를 로드하세요."

    x, y = evt.index[0], evt.index[1]

    if _state["logo_click_start"] is None:
        # 첫 번째 클릭 — 사각형 시작점
        _state["logo_click_start"] = (x, y)
        frame = _state["extracted_frames"].frames[0].copy()
        _draw_logo_rects(frame)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1, cv2.LINE_AA)
        return frame, f"시작점 ({x}, {y}) 선택됨 — 오른쪽 아래를 클릭하세요. (현재 {len(_state['logo_rects'])}개 영역)"
    else:
        # 두 번째 클릭 — 사각형 끝점
        sx, sy = _state["logo_click_start"]
        x1, y1 = min(sx, x), min(sy, y)
        x2, y2 = max(sx, x), max(sy, y)
        _state["logo_rects"].append((x1, y1, x2, y2))
        _state["logo_click_start"] = None

        frame = _state["extracted_frames"].frames[0].copy()
        _draw_logo_rects(frame)
        return frame, f"영역 추가: ({x1},{y1})→({x2},{y2}). 총 {len(_state['logo_rects'])}개 영역. 계속 추가하거나 변환 계산을 진행하세요."


def on_clear_logo_rects():
    """모든 로고 제외 영역을 초기화한다."""
    _state["logo_rects"] = []
    _state["logo_click_start"] = None
    if _state["extracted_frames"] is None:
        return None, "로고 영역 초기화 완료"
    frame = _state["extracted_frames"].frames[0].copy()
    return frame, "로고 영역 초기화 완료 (0개 영역)"



def on_feature_preview(feature_method, max_features, ratio_threshold, enhance_contrast):
    """특징점 검출 및 매칭 결과를 프리뷰한다 (첫 두 프레임 기준)."""
    if _state["seg_results"] is None or len(_state["seg_results"]) < 2:
        return None, None, "먼저 세그멘테이션을 실행하세요 (최소 2프레임)."

    frames = [sr.frame for sr in _state["seg_results"]]
    masks = [sr.mask for sr in _state["seg_results"]]

    if _state["logo_rects"]:
        logo_mask = build_logo_mask(frames[0].shape)
        masks = [cv2.bitwise_or(m, logo_mask) for m in masks]

    # 중간 프레임 기준으로 인접 프레임과 매칭
    ref = len(frames) // 2
    idx_a, idx_b = max(0, ref - 1), ref

    kp_img, match_img, n_kp, n_match = visualize_feature_matches(
        frames[idx_a], frames[idx_b],
        masks[idx_a], masks[idx_b],
        method=feature_method,
        max_features=int(max_features),
        ratio_threshold=ratio_threshold,
        enhance_contrast=enhance_contrast,
    )

    info = (
        f"프레임 #{idx_a} ↔ #{idx_b} | "
        f"특징점: {n_kp}개 | 매칭: {n_match}개 | "
        f"방법: {feature_method} | 대비향상: {'ON' if enhance_contrast else 'OFF'}"
    )
    return kp_img, match_img, info


def on_compute_homography(
    feature_method, transform_type, normalize_zoom,
    max_features, ratio_threshold, enhance_contrast,
):
    """호모그래피를 계산한다."""
    if _state["seg_results"] is None:
        return "먼저 세그멘테이션을 실행하세요."

    frames = [sr.frame for sr in _state["seg_results"]]
    masks = [sr.mask for sr in _state["seg_results"]]

    # 로고/고정 그래픽 영역을 마스크에 추가 (특징점 매칭에서 제외)
    if _state["logo_rects"]:
        logo_mask = build_logo_mask(frames[0].shape)
        masks = [cv2.bitwise_or(m, logo_mask) for m in masks]

    homography_result = compute_cumulative_homographies(
        frames, masks,
        method=feature_method,
        transform_type=transform_type,
        normalize_zoom=normalize_zoom,
        ratio_threshold=ratio_threshold,
        max_features=int(max_features),
        enhance_contrast=enhance_contrast,
    )
    _state["homography_result"] = homography_result

    # 매칭 정보
    lines = [
        f"변환 타입: {homography_result.transform_type}",
        f"기준 프레임: #{homography_result.ref_index}",
        f"특징점: max {int(max_features)} | ratio: {ratio_threshold} | 대비향상: {'ON' if enhance_contrast else 'OFF'}",
    ]
    for i in range(len(frames)):
        if i == homography_result.ref_index:
            lines.append(f"  #{i}: 기준 프레임")
        else:
            lines.append(
                f"  #{i}: 매칭 {homography_result.match_counts[i]}개, "
                f"인라이어 {homography_result.inlier_counts[i]}개"
            )

    return "\n".join(lines)


def on_build_panorama(panorama_method):
    """배경 파노라마를 생성한다."""
    if _state["homography_result"] is None:
        return None, "먼저 호모그래피를 계산하세요."

    frames = [sr.frame for sr in _state["seg_results"]]
    masks = [sr.mask for sr in _state["seg_results"]]
    hr = _state["homography_result"]

    # 로고 영역도 배경에서 제외 (선수 마스크와 합침)
    if _state["logo_rects"]:
        logo_mask = build_logo_mask(frames[0].shape)
        masks = [cv2.bitwise_or(m, logo_mask) for m in masks]

    canvas_w, canvas_h, offset = compute_canvas_size(
        frames[0].shape, hr.cumulative
    )
    canvas_size = (canvas_w, canvas_h)
    _state["canvas_size"] = canvas_size
    _state["offset"] = offset

    pano_result = build_panorama(
        frames, masks, hr.cumulative,
        offset, canvas_size,
        method=panorama_method,
    )
    _state["panorama"] = pano_result.panorama

    logo_info = f", 로고 마스크 {len(_state['logo_rects'])}개 제외" if _state["logo_rects"] else ""
    info = f"파노라마 생성 완료: {canvas_w}×{canvas_h} ({panorama_method}{logo_info})"
    return pano_result.panorama, info


def on_panorama_composite(opacity_mode, edge_feather):
    """파노라마 위에 선수를 합성한다."""
    if _state["panorama"] is None:
        return None, "먼저 파노라마를 생성하세요."

    frames = [sr.frame for sr in _state["seg_results"]]
    masks = [sr.mask for sr in _state["seg_results"]]
    hr = _state["homography_result"]

    result = composite_on_panorama(
        _state["panorama"], frames, masks,
        hr.cumulative, _state["offset"], _state["canvas_size"],
        opacity_mode=opacity_mode,
        edge_feather=int(edge_feather),
    )
    _state["composite_result"] = result

    return result.image, f"파노라마 합성 완료 (투명도: {opacity_mode})"


# ── 4탭: 어노테이션 ──

def on_annotate(show_frame_numbers, show_trajectory, show_height_guide, label_text):
    """합성 이미지에 어노테이션을 추가한다."""
    # 파노라마 합성 결과 또는 단순 합성 결과 사용
    if _state["composite_result"] is not None:
        image = _state["composite_result"].image
        centers = _state["composite_result"].player_centers
    elif _state["seg_results"] is not None:
        # 단순 합성 모드: 마스크 중심점 계산
        from src.compositor import get_mask_center
        image = composite_simple(
            [sr.frame for sr in _state["seg_results"]],
            [sr.mask for sr in _state["seg_results"]],
            opacity_mode="uniform",
        )
        centers = [get_mask_center(sr.mask) for sr in _state["seg_results"]]
    else:
        return None, "먼저 합성을 실행하세요."

    timestamps = None
    if _state["extracted_frames"] is not None:
        timestamps = _state["extracted_frames"].timestamps

    config = AnnotationConfig(
        show_frame_numbers=show_frame_numbers,
        show_trajectory=show_trajectory,
        show_height_guide=show_height_guide,
        show_labels=bool(label_text),
        label_text=label_text,
    )

    annotated = annotate_image(image, centers, config=config, timestamps=timestamps)
    return annotated, "어노테이션 적용 완료"


def save_panorama_result(image):
    """파노라마/어노테이션 결과를 저장한다."""
    if image is None:
        return "저장할 이미지가 없습니다."

    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "composite_advanced.png"

    if isinstance(image, np.ndarray):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    return f"저장 완료: {path}"


# ── Gradio UI ──
with gr.Blocks(title="스포츠 어니언스키닝") as app:
    gr.Markdown("# 스포츠 어니언스키닝 (Motion Composite)")
    gr.Markdown("영상에서 프레임을 추출하고, 선수를 분리하여 하나의 이미지에 합성합니다.")

    # ── 탭 1: 영상 업로드 & 구간 선택 ──
    with gr.Tab("1. 영상 업로드 & 구간 선택"):
        with gr.Row():
            video_input = gr.Video(label="영상 업로드")
            with gr.Column():
                meta_text = gr.Textbox(label="영상 정보", interactive=False)

        with gr.Row():
            start_slider = gr.Slider(
                minimum=0, maximum=1, value=0, step=0.1, label="시작 시점 (초)"
            )
            end_slider = gr.Slider(
                minimum=0, maximum=1, value=1, step=0.1, label="끝 시점 (초)"
            )
            n_frames_slider = gr.Slider(
                minimum=3, maximum=30, value=12, step=1, label="추출 프레임 수"
            )

        extract_btn = gr.Button("프레임 추출", variant="primary")
        extract_info = gr.Textbox(label="추출 정보", interactive=False)
        thumbnail_strip = gr.Image(label="타임라인 썸네일", interactive=False)
        frame_gallery = gr.Gallery(label="추출된 프레임", columns=4, height="auto")

    # ── 탭 2: 세그멘테이션 & 단순 합성 ──
    with gr.Tab("2. 세그멘테이션 & 합성 (고정 카메라)"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 모델 설정")
                yolo_model = gr.Dropdown(
                    choices=["yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolov8x.pt", "yolov8l.pt"],
                    value="yolo11x.pt",
                    label="YOLO 모델",
                )
                sam_model = gr.Dropdown(
                    choices=["sam2.1_l.pt", "sam2.1_b.pt", "sam2_l.pt", "sam2_b.pt", "sam_l.pt"],
                    value="sam2.1_l.pt",
                    label="SAM 모델",
                )
                conf_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="검출 신뢰도 임계값 (어두운 장면은 0.2~0.3 권장)",
                )
                person_idx = gr.Number(
                    value=0, label="대상 선수 인덱스 (0=최고 신뢰도)", precision=0
                )
                enhance_dark = gr.Checkbox(
                    label="어두운 장면 보정 (CLAHE)", value=False,
                )
                segment_btn = gr.Button("세그멘테이션 실행", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 합성 설정")
                opacity_mode = gr.Radio(
                    choices=["uniform", "fade_in", "fade_out"],
                    value="uniform",
                    label="투명도 모드",
                )
                recomposite_btn = gr.Button("재합성")
                save_btn = gr.Button("결과 저장")

        seg_info = gr.Textbox(label="세그멘테이션 정보", interactive=False)

        with gr.Row():
            seg_gallery = gr.Gallery(
                label="세그멘테이션 프리뷰 (마스크 오버레이)",
                columns=4, height="auto",
            )

        gr.Markdown("---")
        gr.Markdown("### 수동 보정")
        gr.Markdown("위 갤러리에서 보정할 프레임을 **클릭**하세요. 아래 이미지에서 **선수 위치를 클릭**하면 SAM이 재세그멘테이션합니다.")

        with gr.Row():
            with gr.Column(scale=2):
                correction_image = gr.Image(
                    label="보정 프레임 (클릭하여 포인트 추가)",
                    interactive=False,
                )
            with gr.Column(scale=1):
                click_mode = gr.Radio(
                    choices=["선수 (전경)", "배경 (제외)"],
                    value="선수 (전경)",
                    label="클릭 모드",
                )
                correction_info = gr.Textbox(label="보정 상태", interactive=False)
                with gr.Row():
                    reset_correction_btn = gr.Button("포인트 초기화")
                    apply_correction_btn = gr.Button("보정 적용 & 재합성", variant="primary")

        gr.Markdown("---")
        composite_output = gr.Image(label="합성 결과", interactive=False)
        save_info = gr.Textbox(label="저장 상태", interactive=False)

    # ── 탭 3: 카메라 모션 대응 (호모그래피 + 파노라마) ──
    with gr.Tab("3. 카메라 모션 대응 (파노라마)"):
        gr.Markdown("### 카메라가 움직이는 영상용 — 호모그래피 정렬 + 배경 파노라마 생성")
        gr.Markdown("*먼저 탭 2에서 세그멘테이션을 완료하세요.*")

        gr.Markdown("#### 로고/고정 그래픽 마스킹")
        gr.Markdown("영상 위의 로고나 자막 등 고정된 그래픽이 있으면 특징점 매칭이 왜곡됩니다. 해당 영역을 사각형으로 지정하여 제외하세요.")

        with gr.Row():
            with gr.Column(scale=2):
                logo_preview = gr.Image(
                    label="로고 영역 지정 (2번 클릭으로 사각형 추가)",
                    interactive=False,
                )
            with gr.Column(scale=1):
                logo_preview_btn = gr.Button("프리뷰 로드", variant="secondary")
                logo_mask_info = gr.Textbox(label="마스킹 상태", interactive=False)
                clear_logo_btn = gr.Button("영역 초기화")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 변환 추정 설정")
                transform_type = gr.Radio(
                    choices=["similarity", "affine", "homography"],
                    value="similarity",
                    label="변환 타입",
                    info="similarity: 팬+줌 (권장) | affine: 팬+줌+전단 | homography: 고정카메라/단순팬",
                )
                feature_method = gr.Radio(
                    choices=["sift", "orb"],
                    value="sift",
                    label="특징점 검출 방법",
                )
                normalize_zoom = gr.Checkbox(
                    label="줌 스케일 정규화 (줌인/아웃 크기 통일)",
                    value=True,
                )

            with gr.Column(scale=1):
                gr.Markdown("#### 매칭 파라미터")
                max_features = gr.Slider(
                    minimum=500, maximum=20000, value=5000, step=500,
                    label="최대 특징점 수 (높을수록 정확, 느림)",
                )
                ratio_threshold = gr.Slider(
                    minimum=0.5, maximum=0.95, value=0.75, step=0.05,
                    label="매칭 비율 임계값 (낮을수록 엄격한 매칭)",
                )
                enhance_contrast = gr.Checkbox(
                    label="CLAHE 대비 향상 (경계선이 뚜렷한 장면에 효과적)",
                    value=False,
                )

        feature_preview_btn = gr.Button("특징점 프리뷰 (매칭 확인)", variant="secondary")
        feature_preview_info = gr.Textbox(label="프리뷰 정보", interactive=False)
        with gr.Row():
            feature_kp_preview = gr.Image(label="검출된 특징점", interactive=False)
            feature_match_preview = gr.Image(label="매칭 결과", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            homography_btn = gr.Button("변환 계산", variant="primary")
        homography_info = gr.Textbox(
            label="변환 매칭 정보", interactive=False, lines=10
        )

        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 파노라마 설정")
                panorama_method = gr.Radio(
                    choices=["average", "median"],
                    value="average",
                    label="배경 합성 방법",
                )
                panorama_btn = gr.Button("파노라마 생성", variant="primary")
                panorama_info = gr.Textbox(label="파노라마 정보", interactive=False)

        panorama_output = gr.Image(label="배경 파노라마", interactive=False)

        gr.Markdown("---")
        gr.Markdown("#### 파노라마 합성")
        with gr.Row():
            pano_opacity_mode = gr.Radio(
                choices=["uniform", "fade_in", "fade_out", "center_focus"],
                value="uniform",
                label="투명도 모드",
            )
            pano_edge_feather = gr.Slider(
                minimum=0, maximum=15, value=3, step=1,
                label="마스크 경계 페더링",
            )
        pano_composite_btn = gr.Button("파노라마 합성 실행", variant="primary")
        pano_composite_info = gr.Textbox(label="합성 정보", interactive=False)
        pano_composite_output = gr.Image(label="파노라마 합성 결과", interactive=False)

    # ── 탭 4: 어노테이션 ──
    with gr.Tab("4. 어노테이션"):
        gr.Markdown("### 합성 이미지에 어노테이션 추가")
        gr.Markdown("*먼저 탭 2 또는 탭 3에서 합성을 완료하세요.*")

        with gr.Row():
            with gr.Column():
                ann_frame_numbers = gr.Checkbox(label="프레임 번호 표시", value=True)
                ann_trajectory = gr.Checkbox(label="궤적선 표시", value=True)
                ann_height_guide = gr.Checkbox(label="높이 가이드 표시", value=False)
                ann_label_text = gr.Textbox(
                    label="라벨 텍스트 (예: Cab Double Cork 1080)",
                    placeholder="트릭 이름 입력...",
                )

        annotate_btn = gr.Button("어노테이션 적용", variant="primary")
        annotate_info = gr.Textbox(label="어노테이션 정보", interactive=False)
        annotated_output = gr.Image(label="어노테이션 결과", interactive=False)

        with gr.Row():
            save_advanced_btn = gr.Button("결과 저장")
            save_advanced_info = gr.Textbox(label="저장 상태", interactive=False)

    # ── 이벤트 연결 ──

    # 탭 1
    video_input.change(
        fn=on_video_upload,
        inputs=[video_input],
        outputs=[meta_text, start_slider, end_slider],
    )
    extract_btn.click(
        fn=on_extract_frames,
        inputs=[start_slider, end_slider, n_frames_slider],
        outputs=[thumbnail_strip, frame_gallery, extract_info],
    )

    # 탭 2
    segment_btn.click(
        fn=on_segment,
        inputs=[yolo_model, sam_model, conf_threshold, person_idx, enhance_dark],
        outputs=[seg_gallery, composite_output, seg_info],
    )
    recomposite_btn.click(
        fn=on_recomposite,
        inputs=[opacity_mode],
        outputs=[composite_output, seg_info],
    )
    save_btn.click(
        fn=save_result,
        inputs=[composite_output],
        outputs=[save_info],
    )
    seg_gallery.select(
        fn=on_select_frame_for_correction,
        outputs=[correction_image, correction_info],
    )
    correction_image.select(
        fn=on_click_correction,
        inputs=[click_mode],
        outputs=[correction_image, correction_info],
    )
    reset_correction_btn.click(
        fn=on_reset_correction,
        outputs=[correction_image, correction_info],
    )
    apply_correction_btn.click(
        fn=on_apply_correction,
        outputs=[seg_gallery, composite_output, seg_info],
    )

    # 탭 3: 로고 마스킹
    logo_preview_btn.click(
        fn=on_logo_preview,
        outputs=[logo_preview, logo_mask_info],
    )
    logo_preview.select(
        fn=on_logo_click,
        outputs=[logo_preview, logo_mask_info],
    )
    clear_logo_btn.click(
        fn=on_clear_logo_rects,
        outputs=[logo_preview, logo_mask_info],
    )

    # 탭 3: 특징점 프리뷰
    feature_preview_btn.click(
        fn=on_feature_preview,
        inputs=[feature_method, max_features, ratio_threshold, enhance_contrast],
        outputs=[feature_kp_preview, feature_match_preview, feature_preview_info],
    )

    # 탭 3: 호모그래피 & 파노라마
    homography_btn.click(
        fn=on_compute_homography,
        inputs=[feature_method, transform_type, normalize_zoom,
                max_features, ratio_threshold, enhance_contrast],
        outputs=[homography_info],
    )
    panorama_btn.click(
        fn=on_build_panorama,
        inputs=[panorama_method],
        outputs=[panorama_output, panorama_info],
    )
    pano_composite_btn.click(
        fn=on_panorama_composite,
        inputs=[pano_opacity_mode, pano_edge_feather],
        outputs=[pano_composite_output, pano_composite_info],
    )

    # 탭 4
    annotate_btn.click(
        fn=on_annotate,
        inputs=[ann_frame_numbers, ann_trajectory, ann_height_guide, ann_label_text],
        outputs=[annotated_output, annotate_info],
    )
    save_advanced_btn.click(
        fn=save_panorama_result,
        inputs=[annotated_output],
        outputs=[save_advanced_info],
    )


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
