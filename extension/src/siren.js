/**
 * 사이렌 — YAMNet 기존 클래스에 임계값만 (#35 11.23 등록 · 결과, `eval/siren.py` #62). **학습하지 않는다.**
 *
 *   오디오 탭(지연 전) ─ 0.48초마다 패치 하나(0.975초) ─▶ offscreen YAMNet ─▶ 4개 사이렌 클래스 최댓값 ≥ θ
 *     ─▶ 트리거 [s, s + 0.975)  — 화면에는 3초 뒤에 닿는다
 *
 * 정의는 eval과 같다 (#62 리뷰에서 창 경계 확인: 패치 k = x[7680·k : 7680·k + 15600]).
 *   - 패치 간격 0.48초 · 창 0.975초 · 패치 창을 그대로 가린다
 *   - 점수 = 390 Siren · 317 Police car · 318 Ambulance · 319 Fire engine 중 최댓값
 *   - θ = 0.0025 — neutral-dev 가림 5% (#35 11.23). 아주 낮아서 브라우저 쪽 점수 차이에 민감하다(#62 리뷰 🟡2) —
 *     브라우저 경로(JS 리샘플러)로 잰 neutral 가림을 README에 따로 적는다
 *
 * 격자의 기준은 붙인 순간(또는 시크 뒤)의 `floor(t / 0.48) · 0.48`이다. eval은 클립 시작이 기준이지만, 기준이 onset과
 * 무관하니 지연 분포는 같다(#62 리뷰).
 *
 * **판정 못 하면 가리지 않는다.** 사이렌은 원래 없던 것을 더하는 경로라, 패치를 못 얻거나 YAMNet이 없으면 그냥 지나간다.
 *
 * 둘로 나눈다.
 *   SB.sirenDecide(scores)  — 순수 판정. 브라우저 없이 테스트한다
 *   SB.SirenSampler         — 오디오 탭에서 0.48초마다 패치를 받아 판정하고 트리거를 낸다
 */
window.SB = window.SB || {};

SB.SIREN_IDX = [390, 317, 318, 319];   // eval/siren.py CLASSES와 같다 — 391(공습경보)은 넣지 않는다

/** @returns {{hit: boolean, p: number}} */
SB.sirenDecide = (scores, theta = SB.config.SIREN_THETA) => {
  let p = -Infinity;
  for (const i of SB.SIREN_IDX) if (scores[i] > p) p = scores[i];
  return { hit: p >= theta, p };
};

SB.SirenSampler = class {
  /**
   * @param {HTMLVideoElement} video
   * @param {object} player   addTriggers를 가진 재생기
   * @param {() => any} getYamnet  지금의 SB.yamnet(없으면 null) — scoresAt(t) → Float32Array(521) | null
   */
  constructor(video, player, getYamnet, cfg = SB.config) {
    this.video = video;
    this.player = player;
    this.getYamnet = getYamnet;
    this.cfg = cfg;
    this.hop = cfg.SIREN_HOP_S;
    this.win = cfg.YAMNET_WIN_S ?? 0.975;
    this.running = false;
    this.next = null;
    this.n = { 패치: 0, 사이렌: 0, 못얻음: 0, 늦음: 0, 다시맞춤: 0 };
    this.round = new SB.Samples();     // scoresAt 요청 → 점수 (패치가 찰 때까지 기다린 시간 포함)
    this.beyond = new SB.Samples();    // 점수를 받은 순간 − 창 끝 (초). eval의 「창 끝 + 0.02 s」에 브라우저가 더하는 몫
    this.slack = new SB.Samples();     // 점수를 받은 순간, 그 패치가 화면에 닿기까지 남은 시간 (초)
    this.recent = [];
  }

  start() {
    this.running = true;
    this._anchor();
    this._loop();
    SB.log(`사이렌 시작 — ${this.hop}초마다 · θ ${this.cfg.SIREN_THETA}`);
  }

  /** 격자를 지금 위치로 다시 맞춘다 — 붙일 때 · 시크 · 너무 밀렸을 때 */
  _anchor() {
    this.next = Math.floor(this.video.currentTime / this.hop) * this.hop;
  }

  async _loop() {
    while (this.running) {
      const y = this.getYamnet();
      if (!y || this.video.paused) { await SB.sleep(250); continue; }
      const live = this.video.currentTime;
      // 이미 화면을 지나간 패치는 건너뛴다 — 탭 숨김 · 오래 멈춤 뒤에 밀린 것을 따라잡느라 늦는 것보다 낫다
      if (this.next + this.win + this.cfg.DELAY_SEC < live || this.next > live + 1) {
        this._anchor();
        this.n.다시맞춤++;
      }
      const s = this.next;
      const t0 = performance.now();
      const scores = await y.scoresAt(s);
      if (!this.running) return;
      if (!scores) {
        this.n.못얻음++;
        this._anchor();                  // 시크 · 원점 변경 — 지금 위치에서 다시
        await SB.sleep(100);
        continue;
      }
      this.round.push(performance.now() - t0);
      this.n.패치++;
      const now = this.video.currentTime;
      this.beyond.push(now - (s + this.win));
      const left = s + this.cfg.DELAY_SEC - now;
      this.slack.push(left);
      if (left < 0) this.n.늦음++;
      const d = SB.sirenDecide(scores, this.cfg.SIREN_THETA);
      this.recent.push({ t: +s.toFixed(2), p: +d.p.toFixed(4), hit: d.hit });
      if (this.recent.length > 20) this.recent.shift();
      if (d.hit) {
        this.n.사이렌++;
        this.player.addTriggers([{
          time: s, category: 'siren', confidence: Math.min(1, d.p), duration: this.win, source: 'yamnet-siren',
        }]);
      }
      this.next = s + this.hop;
    }
  }

  report() {
    return {
      간격_s: this.hop,
      theta: this.cfg.SIREN_THETA,
      ...this.n,
      왕복_ms: this.round.summary(),
      창끝_뒤_s: this.beyond.summary(),     // eval 지연(창 끝 − onset)에 브라우저가 더하는 몫
      화면까지_남은_s: this.slack.summary(),
      ...(this.running ? {} : { 상태: '꺼짐' }),
      최근: this.recent,
    };
  }

  stop() { this.running = false; }
};

SB.sleep = SB.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
