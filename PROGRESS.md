# 스포츠 어니언스키닝 — 진행 상황

## 현재 상태: Phase 1 MVP 진행 중

### 구현 완료

#### 0단계: 영상 업로드 & 구간 선택 ✅
- `src/clip_selector.py`
- 영상 메타데이터 자동 읽기 (FPS, 해상도, 코덱, 길이)
- 시작/끝 시점 기반 구간 선택 (`ClipSelection`)
- 타임라인 썸네일 스트립 생성
- 특정 시점 프리뷰 프레임 반환

#### 1단계: 프레임 추출 ✅
- `src/frame_extractor.py`
- 균등 간격 추출 (`extract_frames_uniform`) — N개 프레임을 균등 분배
- 시간 간격 추출 (`extract_frames_by_interval`) — 초 단위 간격
- 전체 프레임 추출 (`extract_all_frames`)
- 프레임 파일 저장 (`save_frames`)

#### 4단계: 선수 세그멘테이션 ✅
- `src/segmentation.py`
- YOLO (v8/v11) 기반 사람 검출 (class 0 = person)
- SAM2/SAM2.1 bbox 프롬프트 기반 정밀 마스크 생성
- 단일/일괄 프레임 세그멘테이션
- 마스크 오버레이 시각화 (`apply_mask_overlay`)
- RGBA 선수 추출 (`extract_player`)

#### 파이프라인 & UI ✅
- `src/pipeline.py` — 0→1→4단계 통합, 단순 합성 (고정 카메라 전용)
- `app.py` — Gradio 웹 UI (2개 탭)
  - 탭 1: 영상 업로드, 구간 슬라이더, 프레임 추출 갤러리
  - 탭 2: 모델 설정, 세그멘테이션 실행, 합성 결과 출력
- 투명도 모드 3종: uniform / fade_in / fade_out

---

### 미구현 (TODO)

#### 2단계: 카메라 모션 추정 ⬜
- 호모그래피 기반 프레임 정렬 (`src/homography.py`)
- SIFT/ORB 특징점 매칭 + RANSAC
- 선수 마스크 제외 후 배경 특징점만 매칭

#### 3단계: 배경 파노라마 생성 ⬜
- 워핑된 프레임의 중앙값 기반 배경 합성 (`src/panorama.py`)
- Multi-band blending

#### 5단계: 합성 & 블렌딩 (고급) ⬜
- 파노라마 좌표계 기반 합성 (`src/compositor.py`)
- Poisson blending
- 궤적선 오버레이

#### 6단계: 어노테이션 ⬜
- 높이 눈금, 트릭 이름, 회전 수 라벨 (`src/annotator.py`)

---

### 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| 영상 처리 | OpenCV 4.13 |
| 객체 검출 | YOLO v11 (ultralytics 8.4) |
| 세그멘테이션 | SAM 2.1 (ultralytics) |
| 딥러닝 | PyTorch 2.10 |
| 웹 UI | Gradio 6.5 |
| 언어 | Python 3.10 |

### 샘플 데이터

- `examples/sample.mov` — 테스트용 화면 녹화 (2288×1290, 41fps, 18.9초)
- `SCR-20260213-liwh.jpeg` — 참고 이미지 (NYT Chloe Kim 스타일)

---

*마지막 업데이트: 2026-02-13*
