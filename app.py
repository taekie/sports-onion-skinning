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
from src.segmentation import PlayerSegmenter, apply_mask_overlay
from src.pipeline import composite_simple, PipelineConfig

# ── 전역 상태 ──
_state = {
    "video_path": None,
    "meta": None,
    "extracted_frames": None,
    "seg_results": None,
    "segmenter": None,
}


def on_video_upload(video_path):
    """영상 업로드 시 메타데이터를 읽고 슬라이더 범위를 설정한다."""
    if video_path is None:
        return (
            "영상을 업로드하세요.",
            gr.update(maximum=1, value=0),
            gr.update(maximum=1, value=1),
            None,
        )

    meta = get_video_meta(video_path)
    _state["video_path"] = video_path
    _state["meta"] = meta

    # 썸네일 스트립 생성
    thumbs = generate_thumbnail_strip(video_path, n_thumbnails=8)
    if thumbs:
        strip = np.concatenate(thumbs, axis=1)
    else:
        strip = None

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
        strip,
    )


def on_extract_frames(start_sec, end_sec, n_frames):
    """구간 선택 후 프레임을 추출하고 갤러리에 표시한다."""
    if _state["video_path"] is None or _state["meta"] is None:
        return [], "먼저 영상을 업로드하세요."

    meta = _state["meta"]
    clip = make_clip_selection(meta, start_sec, end_sec)
    extracted = extract_frames_uniform(_state["video_path"], clip, meta, n_frames=int(n_frames))
    _state["extracted_frames"] = extracted

    info = (
        f"구간: {clip.start_sec:.1f}s ~ {clip.end_sec:.1f}s ({clip.duration_sec:.1f}초)\n"
        f"추출 프레임: {len(extracted.frames)}개"
    )

    gallery_images = [(frame, f"#{i} ({ts:.2f}s)") for i, (frame, ts)
                      in enumerate(zip(extracted.frames, extracted.timestamps))]

    return gallery_images, info


def on_segment(yolo_model, sam_model, conf_threshold, person_idx):
    """추출된 프레임들에 대해 세그멘테이션을 실행한다."""
    if _state["extracted_frames"] is None:
        return [], None, "먼저 프레임을 추출하세요."

    segmenter = PlayerSegmenter(
        yolo_model=yolo_model,
        sam_model=sam_model,
        person_conf_threshold=conf_threshold,
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


# ── Gradio UI ──
with gr.Blocks(title="스포츠 어니언스키닝") as app:
    gr.Markdown("# 스포츠 어니언스키닝 (Motion Composite)")
    gr.Markdown("영상에서 프레임을 추출하고, 선수를 분리하여 하나의 이미지에 합성합니다.")

    with gr.Tab("1. 영상 업로드 & 구간 선택"):
        with gr.Row():
            video_input = gr.Video(label="영상 업로드")
            with gr.Column():
                meta_text = gr.Textbox(label="영상 정보", interactive=False)
                thumbnail_strip = gr.Image(label="타임라인 썸네일", interactive=False)

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
        frame_gallery = gr.Gallery(label="추출된 프레임", columns=4, height="auto")

    with gr.Tab("2. 세그멘테이션 & 합성"):
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
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="검출 신뢰도 임계값",
                )
                person_idx = gr.Number(
                    value=0, label="대상 선수 인덱스 (0=최고 신뢰도)", precision=0
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

        composite_output = gr.Image(label="합성 결과", interactive=False)
        save_info = gr.Textbox(label="저장 상태", interactive=False)

    # ── 이벤트 연결 ──
    video_input.change(
        fn=on_video_upload,
        inputs=[video_input],
        outputs=[meta_text, start_slider, end_slider, thumbnail_strip],
    )

    extract_btn.click(
        fn=on_extract_frames,
        inputs=[start_slider, end_slider, n_frames_slider],
        outputs=[frame_gallery, extract_info],
    )

    segment_btn.click(
        fn=on_segment,
        inputs=[yolo_model, sam_model, conf_threshold, person_idx],
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


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
