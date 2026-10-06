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
};

SB.log = (...a) => console.log('[scareblock]', ...a);
