/**
 * 설정 화면 — 툴바 팝업 · 옵션 페이지 · 설치 직후 안내 탭이 같은 파일이다(`settings.html`).
 * 값은 chrome.storage.local `settings` 하나에 저장하고, 열려 있는 유튜브 탭에는 중계(bridge.js)가 바로 넘긴다.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const welcomeTab = new URLSearchParams(location.search).has('welcome');
  if (welcomeTab) document.body.classList.add('page');

  const MODE_HELP = {
    blur: '그 장면을 흐리게 가리고 소리를 줄여요.',
    warn: '가리지 않아요. 화면에 닿기 1.5초 전부터 「⚠ 곧 …」을 위에 띄워요.',
  };
  const SENS_LABEL = { 1: '1 · 가장 덜 민감', 2: '2', 3: '3 · 기본', 4: '4', 5: '5 · 가장 민감' };

  let s = SB.normalizeSettings(null);
  let statusTimer = null;

  function render() {
    $('enabled').checked = s.enabled;
    $('opts').disabled = !s.enabled;
    for (const r of document.querySelectorAll('input[name=mode]')) r.checked = r.value === s.mode;
    $('modeHelp').textContent = MODE_HELP[s.mode];
    $('sensitivity').value = String(s.sensitivity);
    $('sensVal').textContent = SENS_LABEL[s.sensitivity];
    for (const k of Object.keys(s.cats)) $(`cat-${k}`).checked = s.cats[k];
    $('welcome').hidden = !(welcomeTab || !s.onboarded);
  }

  async function save(patch) {
    s = SB.normalizeSettings({ ...s, ...patch, cats: { ...s.cats, ...(patch.cats || {}) } });
    render();
    try {
      await chrome.storage.local.set({ settings: s });
      flash('저장됨');
    } catch (e) {
      flash(`저장 못 함 — ${e.message}`);
    }
  }

  function flash(text) {
    $('status').textContent = text;
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => { $('status').textContent = ''; }, 1500);
  }

  $('enabled').addEventListener('change', (e) => save({ enabled: e.target.checked }));
  for (const r of document.querySelectorAll('input[name=mode]')) {
    r.addEventListener('change', (e) => { if (e.target.checked) save({ mode: e.target.value }); });
  }
  // 끌어서 바꾸는 동안에는 글자만 바꾸고, 놓았을 때 저장한다 — 탭에 설정이 계속 날아가지 않게
  $('sensitivity').addEventListener('input', (e) => { $('sensVal').textContent = SENS_LABEL[e.target.value]; });
  $('sensitivity').addEventListener('change', (e) => save({ sensitivity: Number(e.target.value) }));
  for (const k of Object.keys(SB.SETTINGS_DEFAULT.cats)) {
    $(`cat-${k}`).addEventListener('change', (e) => save({ cats: { [k]: e.target.checked } }));
  }
  $('gotit').addEventListener('click', () => {
    save({ onboarded: true });
    if (welcomeTab) window.close();
  });

  chrome.storage.local.get('settings').then(({ settings }) => {
    s = SB.normalizeSettings(settings);
    render();
  }, () => render());
})();
