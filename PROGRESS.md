# 스포츠 어니언스키닝 — 진행 상황

## 현재 상태: 전체 파이프라인 구현 완료 + UI 개선

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

#### 2단계: 카메라 모션 추정 ✅
- `src/homography.py`
- SIFT/ORB 특징점 검출 (`detect_and_compute`) — 선수 마스크 + 로고 마스크 제외
- CLAHE 대비 향상 옵션 — 경계선이 뚜렷한 장면(하프파이프 등) 특징점 검출 개선
- Lowe's ratio test 기반 특징점 매칭 (`match_features`)
- 변환 추정 3종 (`estimate_transform`):
  - `similarity` (4DOF): 이동+회전+균일스케일 — 팬+줌에 최적 (기본값)
  - `affine` (6DOF): 이동+회전+비균일스케일+전단
  - `homography` (8DOF): 완전 사영 변환
- 인접 프레임 간 순차 변환 (`compute_pairwise_homographies`)
- 기준 프레임 중심 누적 변환 (`compute_cumulative_homographies`)
- 줌 스케일 정규화 (`normalize_scales`) — 줌인/아웃 크기 통일
- 캔버스 크기 & 오프셋 자동 계산 (`compute_canvas_size`)
- 특징점 매칭 시각화 (`visualize_feature_matches`) — 디버깅용 프리뷰

#### 3단계: 배경 파노라마 생성 ✅
- `src/panorama.py`
- 프레임/마스크 호모그래피 워핑 (`warp_frame`, `warp_mask`)
- 커버리지 마스크 계산 (`get_warp_coverage_mask`)
- 중앙값(median) 기반 배경 합성 — 움직이는 객체 자동 제거
- 가중 평균(average) 기반 배경 합성 — 빠르고 메모리 효율적
- 로고/고정 그래픽 영역 제외 후 배경 합성
- 통합 인터페이스 (`build_panorama`)

#### 4단계: 선수 세그멘테이션 ✅
- `src/segmentation.py`
- YOLO (v8/v11) 기반 사람 검출 (class 0 = person)
- SAM2/SAM2.1 bbox 프롬프트 기반 정밀 마스크 생성
- SAM 포인트 프롬프트 기반 수동 보정 (`segment_with_points`)
- 어두운 장면 CLAHE 전처리 (`enhance_frame_clahe`)
- 단일/일괄 프레임 세그멘테이션
- 마스크 오버레이 시각화 (`apply_mask_overlay`)
- RGBA 선수 추출 (`extract_player`)

#### 5단계: 합성 & 블렌딩 (고급) ✅
- `src/compositor.py`
- 파노라마 좌표계 기반 선수 합성 (`composite_on_panorama`)
- 투명도 모드 4종: uniform / fade_in / fade_out / center_focus
- 마스크 경계 페더링 (GaussianBlur 기반 부드러운 합성)
- 선수 무게중심 추적 (`get_mask_center`)
- 이동 궤적선 그리기 (`draw_trajectory`)
- 색상 틴트 적용 (`apply_color_tint`)
- 드롭 섀도우 추가 (`add_shadow`)

#### 6단계: 어노테이션 ✅
- `src/annotator.py`
- 프레임 번호 & 타임스탬프 라벨 표시 (`annotate_frame_numbers`)
- 궤적선 3종 스타일: line / curve / dashed (`annotate_trajectory`)
- 높이 가이드라인 — 상대 높이 눈금 표시 (`annotate_height_guide`)
- 텍스트 라벨 (트릭 이름 등) — 4방향 배치 (`annotate_label`)
- 통합 어노테이션 적용 (`annotate_image`)

#### 파이프라인 & UI ✅
- `src/pipeline.py` — 전체 6단계 통합
  - 모드 1 (고정 카메라): 0→1→4→단순합성
  - 모드 2 (카메라 모션): 0→1→4→2→3→5→6
- `app.py` — Gradio 웹 UI (4개 탭)
  - 탭 1: 영상 업로드, 구간 슬라이더, 프레임 추출 갤러리
  - 탭 2: 모델 설정, 세그멘테이션, 수동 보정 (SAM 포인트 프롬프트), 단순 합성
  - 탭 3: 로고 마스킹, 특징점 프리뷰, 매칭 파라미터 튜닝, 호모그래피 계산, 배경 파노라마, 파노라마 합성
  - 탭 4: 어노테이션 (프레임 번호, 궤적선, 높이 가이드, 라벨)

#### UI 개선사항
- 프레임 추출: 버튼 클릭 시에만 실행 (자동 추출 제거)
- 어두운 장면 보정: CLAHE 전처리 옵션 (체크박스)
- 검출 신뢰도 임계값 조정 (기본 0.3, 어두운 장면용)
- 수동 보정: 갤러리 클릭 → 포인트 프롬프트로 SAM 재실행 (전경/배경 클릭 모드)
- 로고/고정 그래픽 마스킹: 2클릭 사각형으로 제외 영역 지정 (복수 영역 지원)
- 변환 타입 선택: similarity (팬+줌 권장) / affine / homography
- 줌 스케일 정규화 옵션
- 특징점 프리뷰: 검출된 특징점 + 매칭 결과 시각화 (파라미터 조정 전 확인용)
- 매칭 파라미터 튜닝: 최대 특징점 수, 비율 임계값, CLAHE 대비 향상

---

### 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| 영상 처리 | OpenCV 4.13 |
| 객체 검출 | YOLO v11 (ultralytics 8.4) |
| 세그멘테이션 | SAM 2.1 (ultralytics) |
| 딥러닝 | PyTorch 2.10 |
| 웹 UI | Gradio 6.5 |
| 이미지 처리 | Pillow |
| 언어 | Python 3.10 |

### 샘플 데이터

- `examples/sample.mov` — 테스트용 화면 녹화 (2288×1290, 41fps, 18.9초)
- `SCR-20260213-liwh.jpeg` — 참고 이미지 (NYT Chloe Kim 스타일)

---

*마지막 업데이트: 2026-02-13*
