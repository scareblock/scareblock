/**
 * 사용자 설정 — 04단계 개인화. 페이지(MAIN) · 설정 화면이 같이 쓰는 순수 계산이다.
 *
 * 저장은 `chrome.storage.local`의 `settings` 하나. 페이지는 `chrome.*`가 없어서 중계(`bridge.js`)가
 * 읽어 window 메시지로 넘긴다(`infer-client.js`와 같은 길). 값이 이상하거나 없으면 기본값으로 돈다.
 *
 * **기본값 = 지금까지의 동작.** 설정을 한 번도 안 건드린 사용자는 E1과 같은 탐지기를 쓴다.
 */
window.SB = window.SB || {};

SB.SETTINGS_TAG = 'scareblock-settings';

/** 화면에 보이는 이름. 계약의 category 값 → 한국어 */
SB.CATEGORY_LABEL = {
  jumpscare: '점프 스케어',
  blood: '피',
  spider: '거미',
  syringe: '주사기',
  siren: '사이렌',
  manual: '수동',
};

/**
 * 민감도 1~5 → 규칙 탐지기 surge_of_quiet. 높을수록 작은 급등에도 반응한다.
 * **3 = 20 = E1 · #35 사전 등록 동작점.** 데모용 5(#46)는 가장 민감한 쪽이다.
 * 단계는 발화율을 **잰 값**에 맞췄다(#61 리뷰) — `eval/README.md` 「동작점 고르기」의 라벨 없는 스윕 surge 5 · 10 · 20 · 35 · 60.
 * 4(surge 10)는 #65 리뷰에서 쟀다 — 이제 다섯 단계가 다 잰 값이다.
 */
SB.SENSITIVITY_SURGE = { 1: 60, 2: 35, 3: 20, 4: 10, 5: 5 };

/** 단계마다 공포 영상(44클립 · 212.7분, 2026.09.15)에서 1분에 울린 횟수 — 라벨 없이 잰 값. 설정 화면이 보여 준다 */
SB.SENSITIVITY_RATE = { 1: 0.49, 2: 1.18, 3: 2.19, 4: 3.94, 5: 6.37 };

SB.SETTINGS_DEFAULT = Object.freeze({
  enabled: true,
  mode: 'blur',            // 'blur' — 가리고 소리를 줄인다 | 'warn' — 가리지 않고 미리 알려 준다
  sensitivity: 3,
  cats: Object.freeze({ jumpscare: true, blood: true, spider: true, syringe: true, siren: true }),
  onboarded: false,
});

/** 저장소에서 온 값을 믿지 않는다 — 모르는 키는 버리고, 범위 밖은 기본값으로 */
SB.normalizeSettings = (raw) => {
  const d = SB.SETTINGS_DEFAULT;
  const r = raw && typeof raw === 'object' ? raw : {};
  const lvl = Number(r.sensitivity);
  const cats = {};
  for (const k of Object.keys(d.cats)) {
    cats[k] = typeof r.cats?.[k] === 'boolean' ? r.cats[k] : d.cats[k];
  }
  return {
    enabled: typeof r.enabled === 'boolean' ? r.enabled : d.enabled,
    mode: r.mode === 'warn' || r.mode === 'blur' ? r.mode : d.mode,
    sensitivity: Number.isInteger(lvl) && lvl >= 1 && lvl <= 5 ? lvl : d.sensitivity,
    cats,
    onboarded: typeof r.onboarded === 'boolean' ? r.onboarded : d.onboarded,
  };
};

/** CLIP 종류를 하나라도 켰을 때만 CLIP을 돌린다 — 다 끄면 1초마다 돌고 버리기만 한다(#61 리뷰) */
SB.clipWanted = (s, cfg = SB.config) => !!cfg.CLIP && (cfg.CLIP_CATS || []).some((c) => s.cats?.[c]);

/** 사이렌을 끄면 사이렌 샘플러도 멈춘다 — 0.48초마다 YAMNet을 부르니까(clipWanted와 같은 이유) */
SB.sirenWanted = (s, cfg = SB.config) => !!cfg.SIREN && !!s.cats?.siren;

SB.surgeFor = (level) => SB.SENSITIVITY_SURGE[level] ?? SB.SENSITIVITY_SURGE[SB.SETTINGS_DEFAULT.sensitivity];

/**
 * 돌고 있는 재생기 · 탐지기에 설정을 건다. 켜기/끄기는 부르는 쪽(main.js)이 붙이고 떼는 것으로 한다.
 * @returns {string[]} 바뀐 것 — 로그용
 */
SB.applySettings = (s, { player, detector } = {}) => {
  const changed = [];
  if (player) {
    if (player.mode !== s.mode) { player.mode = s.mode; changed.push(`모드 ${s.mode === 'warn' ? '경고만' : '블러'}`); }
    const off = Object.keys(s.cats).filter((k) => !s.cats[k]);
    const prev = [...(player.disabledCats || [])].sort().join(',');
    if (prev !== [...off].sort().join(',')) {
      player.disabledCats = new Set(off);
      changed.push(off.length ? `끔: ${off.map((k) => SB.CATEGORY_LABEL[k]).join(' · ')}` : '모든 트리거 켬');
    }
  }
  if (detector?.setSurge && detector.core?.p.surge_of_quiet !== SB.surgeFor(s.sensitivity)) {
    detector.setSurge(SB.surgeFor(s.sensitivity));
    changed.push(`민감도 ${s.sensitivity}`);
  }
  return changed;
};
