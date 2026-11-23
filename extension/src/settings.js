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
  manual: '수동',
};

/**
 * 민감도 1~5 → 규칙 탐지기 surge_of_quiet. 높을수록 작은 급등에도 반응한다.
 * **3 = 20 = E1 · #35 사전 등록 동작점.** 데모용 5(#46)는 가장 민감한 쪽이다.
 * 일반 영상에서 자주 울리는 쪽(낮은 값)은 말소리 기각(#52)이 일부를 거른다.
 */
SB.SENSITIVITY_SURGE = { 1: 40, 2: 30, 3: 20, 4: 10, 5: 5 };

SB.SETTINGS_DEFAULT = Object.freeze({
  enabled: true,
  mode: 'blur',            // 'blur' — 가리고 소리를 줄인다 | 'warn' — 가리지 않고 미리 알려 준다
  sensitivity: 3,
  cats: Object.freeze({ jumpscare: true, blood: true, spider: true, syringe: true }),
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
