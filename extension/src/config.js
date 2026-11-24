/** 전역 설정. 값의 근거는 poc/README.md 「720p 자원 측정」에 있다. */
window.SB = window.SB || {};

SB.config = {
  DELAY_SEC: 3.0,        // 지연 버퍼 길이
  CAPTURE_FPS: 40,       // 60fps도 가능하나 메모리(190슬롯 668 MB) 때문에 40으로 둔다
  BUF_W: 1280,
  BUF_H: 720,
  BLUR_PX: 28,
  STALL_MS: 32,          // 렌더 끊김 판정 (60Hz 두 프레임)
  FADE_DB: 0.25,         // 블러 중 음량 배율
  DETECTOR: 'rule',      // 'rule' | 'fake' — fake는 10초마다 가짜 트리거 (M1 계측용)
  TRIGGER_TAIL_S: 1.5,   // 순간 사건 + 꼬리 — README 인터페이스 계약
  WARN_LEAD_S: 1.5,      // 경고만 모드: 화면에 닿기 이만큼 전부터 「곧 …」을 띄운다 (지연 3초 안)
  VISUAL_FPS: 2,         // 시각 특징 샘플링 — CLIP과 같은 1~2fps 전제 (README 「모델이 아닌 것」)
  YAMNET: true,          // 별도 문서 추론. offscreen이 준비됐다고 답할 때만 오디오 탭을 붙인다 — 모델이 없으면 비용 0
  SPEECH_GATE: true,     // 말소리 기각(#40). YAMNet이 없으면 판정 없이 그대로 가린다
  CLIP: true,            // 피 · 거미 · 주사기. offscreen에 CLIP이 올라왔을 때만 샘플러를 붙인다
  CLIP_FPS: 1,           // eval/clip_zeroshot과 같은 1초에 한 장
  CLIP_CATS: ['blood', 'spider', 'syringe'],
  // θ — neutral-dev 프레임의 5%만 가리는 값 (#35 04단계 · 라벨 보기 전 고정). 바꾸면 보고한 숫자와 달라진다
  CLIP_THETA: { blood: 0.2490658164024353, spider: 0.03803376853466034, syringe: 0.0349690280854702 },
  // 피 보강 모델(#53)의 θ — 같은 5% 규칙. models/clip_heads.json에 보강 모델이 있을 때만 쓴다
  // 리사이즈를 버림으로 고친 뒤(#51 리뷰 1) 다시 학습한 값 (11.22 게시 0.05835 → 0.05868, _local/dataset/resize-1123)
  CLIP_PROBE_THETA: { blood: 0.05867912247776985 },
};

SB.log = (...a) => console.log('[scareblock]', ...a);
