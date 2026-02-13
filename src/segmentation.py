"""4단계: 선수 세그멘테이션 (Player Segmentation)

YOLO로 사람(선수) 바운딩 박스를 검출하고,
SAM 2로 정밀 세그멘테이션 마스크를 생성한다.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SegmentationResult:
    """단일 프레임의 세그멘테이션 결과"""
    frame: np.ndarray          # 원본 프레임 (RGB)
    mask: np.ndarray           # 바이너리 마스크 (H, W), 0 또는 255
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) 또는 None
    confidence: float          # 검출 신뢰도


def enhance_frame_clahe(frame: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8) -> np.ndarray:
    """CLAHE(적응형 히스토그램 균일화)로 어두운 프레임의 대비를 개선한다.

    Args:
        frame: RGB 이미지
        clip_limit: 대비 제한 값 (높을수록 더 강한 보정)
        grid_size: 타일 그리드 크기

    Returns:
        대비가 개선된 RGB 이미지
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


class PlayerSegmenter:
    """YOLO + SAM 기반 선수 세그멘테이션 파이프라인"""

    def __init__(
        self,
        yolo_model: str = "yolo11x.pt",
        sam_model: str = "sam2_l.pt",
        device: str = "auto",
        person_conf_threshold: float = 0.5,
        enhance_dark: bool = False,
        clahe_clip_limit: float = 3.0,
    ):
        """
        Args:
            yolo_model: YOLO 모델 이름 (자동 다운로드)
            sam_model: SAM 모델 이름 (자동 다운로드)
            device: 'cpu', 'cuda', 'mps', 'auto'
            person_conf_threshold: 사람 검출 최소 신뢰도
            enhance_dark: 어두운 프레임 대비 보정 (CLAHE) 활성화
            clahe_clip_limit: CLAHE 대비 제한 값
        """
        self.yolo_model_name = yolo_model
        self.sam_model_name = sam_model
        self.device = device
        self.person_conf_threshold = person_conf_threshold
        self.enhance_dark = enhance_dark
        self.clahe_clip_limit = clahe_clip_limit
        self._yolo = None
        self._sam = None

    def _load_models(self):
        """모델을 지연 로딩한다."""
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self.yolo_model_name)
        if self._sam is None:
            from ultralytics import SAM
            self._sam = SAM(self.sam_model_name)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """검출 전 프레임 전처리 (어두운 장면 보정)."""
        if self.enhance_dark:
            return enhance_frame_clahe(frame, clip_limit=self.clahe_clip_limit)
        return frame

    def detect_person(self, frame: np.ndarray) -> list[tuple[list[int], float]]:
        """프레임에서 사람(person, class=0)을 검출한다.

        Returns:
            [(bbox, confidence), ...] 리스트. bbox는 [x1, y1, x2, y2]
        """
        self._load_models()
        enhanced = self._preprocess(frame)
        results = self._yolo(enhanced, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                if cls_id == 0 and conf >= self.person_conf_threshold:  # class 0 = person
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    detections.append((bbox, conf))

        # 신뢰도 높은 순 정렬
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections

    def segment_with_bbox(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        """SAM을 bbox 프롬프트로 실행하여 마스크를 반환한다.

        Args:
            frame: RGB 이미지 (H, W, 3)
            bbox: [x1, y1, x2, y2]

        Returns:
            바이너리 마스크 (H, W), 값 0 또는 255
        """
        self._load_models()
        results = self._sam(frame, bboxes=[bbox], verbose=False)
        return self._extract_mask(results, frame.shape[:2])

    def segment_with_points(
        self,
        frame: np.ndarray,
        points: list[list[int]],
        labels: list[int],
    ) -> np.ndarray:
        """SAM을 포인트 프롬프트로 실행하여 마스크를 반환한다.

        Args:
            frame: RGB 이미지 (H, W, 3)
            points: [[x, y], ...] 클릭 좌표 리스트
            labels: [1, 0, ...] 각 포인트의 라벨 (1=전경/선수, 0=배경)

        Returns:
            바이너리 마스크 (H, W), 값 0 또는 255
        """
        self._load_models()
        results = self._sam(frame, points=points, labels=labels, verbose=False)
        return self._extract_mask(results, frame.shape[:2])

    def _extract_mask(self, results, frame_shape: tuple[int, int]) -> np.ndarray:
        """SAM 결과에서 마스크를 추출한다."""
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            # 0~1 float → 0 또는 255 uint8
            mask = (mask > 0.5).astype(np.uint8) * 255
            # 마스크 크기가 프레임과 다를 경우 리사이즈
            if mask.shape[:2] != frame_shape:
                mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            return mask

        # 마스크 추출 실패 시 빈 마스크
        return np.zeros(frame_shape, dtype=np.uint8)

    def segment_frame(
        self,
        frame: np.ndarray,
        target_person_idx: int = 0,
    ) -> SegmentationResult:
        """단일 프레임에서 선수를 검출 + 세그멘테이션한다.

        Args:
            frame: RGB 이미지
            target_person_idx: 검출된 사람 중 몇 번째를 선택할지 (0 = 가장 신뢰도 높은 사람)
        """
        detections = self.detect_person(frame)

        if not detections:
            return SegmentationResult(
                frame=frame,
                mask=np.zeros(frame.shape[:2], dtype=np.uint8),
                bbox=None,
                confidence=0.0,
            )

        idx = min(target_person_idx, len(detections) - 1)
        bbox, conf = detections[idx]
        mask = self.segment_with_bbox(frame, bbox)

        return SegmentationResult(
            frame=frame,
            mask=mask,
            bbox=tuple(bbox),
            confidence=conf,
        )

    def segment_frames(
        self,
        frames: list[np.ndarray],
        target_person_idx: int = 0,
    ) -> list[SegmentationResult]:
        """여러 프레임을 일괄 세그멘테이션한다."""
        self._load_models()  # 미리 로딩
        results = []
        for frame in frames:
            result = self.segment_frame(frame, target_person_idx)
            results.append(result)
        return results


def extract_player(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크를 이용해 선수 영역만 추출 (RGBA)."""
    rgba = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = frame
    rgba[:, :, 3] = mask
    return rgba


def apply_mask_overlay(frame: np.ndarray, mask: np.ndarray,
                        color: tuple = (0, 120, 255), alpha: float = 0.4) -> np.ndarray:
    """프레임에 마스크를 반투명 오버레이하여 시각화한다."""
    overlay = frame.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) +
        np.array(color, dtype=np.float64) * alpha
    ).astype(np.uint8)
    return overlay
