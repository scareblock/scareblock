/**
 * 말소리 기각 — 사후 검증 (#40 · #35 10.03 사전 등록). `eval/speech_gate.py`(#40 브랜치 `adb83cf` — main에는 아직 없다)의 gate()를 옮긴 것이다.
 *
 * 규칙 탐지가 울리면 사건 시작(onset)부터 YAMNet 패치 하나(0.975초)를 보고,
 *   `Speech ≥ θ(0.1)` **이고** `Speech > #43 등록 20개 클래스의 최댓값` 이면 그 탐지를 버린다.
 * 조건은 **둘 다**다. Speech만 보면 비명 · 고함처럼 사람 목소리가 섞인 점프 스케어까지 버린다.
 *
 * **판정 못 하면 버리지 않는다.** YAMNet이 없거나, 패치를 못 얻었거나, 표시 시각 전에 답이 안 오면
 * 탐지를 그대로 낸다 — 놓치는 것(미탐)이 잘못 가리는 것(오탐)보다 나쁘다(README 「클립 추출 규칙」).
 *
 * 둘로 나눈다.
 *   SB.speechDecide(scores)  — 순수 판정. 브라우저 없이 테스트한다
 *   SB.SpeechGate            — 탐지를 잡아 두었다가 판정 뒤 재생기에 넘긴다
 */
window.SB = window.SB || {};

SB.SPEECH_IDX = 0;   // AudioSet 클래스 0 = Speech
SB.SPEECH_THETA = 0.1;   // #35 10.03 등록 · 개발 세트에서 고른 값
// `eval/yamnet.py` CLASSES와 같은 20개 (#43 등록). 바꾸면 E1 · #40 숫자와 달라진다
SB.SCARE_IDX = [6, 9, 11, 74, 80, 105, 281, 348, 352, 353, 420, 421, 430, 435, 437, 454, 460, 462, 463, 464];

/** @returns {{reject: boolean, speech: number, scare: number}} */
SB.speechDecide = (scores, theta = SB.SPEECH_THETA) => {
  const speech = scores[SB.SPEECH_IDX];
  let scare = -Infinity;
  for (const i of SB.SCARE_IDX) if (scores[i] > scare) scare = scores[i];
  return { reject: speech >= theta && speech > scare, speech, scare };
};

SB.SpeechGate = class {
  /**
   * @param {() => any} getYamnet  지금의 SB.yamnet(없으면 null). 영상이 바뀌면 달라지므로 매번 묻는다
   * @param {(t: object) => void} emit  통과한 트리거를 넘길 곳 (재생기의 addTriggers)
   * @param {number} [theta]  Speech 기준값 (기본 SB.SPEECH_THETA = 0.1, #35 10.03 등록)
   * @param {number} [marginMs]  남은 시간에서 미리 빼 두는 여유(ms). 남은 시간은 submit(trig, msLeft)로 받는다
   */
  constructor(getYamnet, emit, theta = SB.SPEECH_THETA, marginMs = 250) {
    this.getYamnet = getYamnet;
    this.emit = emit;
    this.theta = theta;
    this.marginMs = marginMs;
    this.stats = { 받음: 0, 통과: 0, 기각: 0, 판정없음_통과: 0, 시간초과_통과: 0 };
    this.recent = [];   // 최근 판정 20개 — 확인용
  }

  /**
   * 트리거 하나를 판정한다. 끝나면 통과한 경우에만 emit을 부른다.
   * @param {object} trig   계약 트리거 (time = onset)
   * @param {number} msLeft  화면에 닿기까지 남은 시간. 이 안에 답이 없으면 그냥 낸다
   * @returns {Promise<'keep'|'reject'|'unseen'|'timeout'>}
   */
  async submit(trig, msLeft) {
    this.stats.받음++;
    const y = this.getYamnet();
    if (!y) return this._pass(trig, 'unseen');
    const budget = msLeft - this.marginMs;
    if (!(budget > 0)) return this._pass(trig, 'timeout');

    let timer;
    const late = new Promise((res) => { timer = setTimeout(() => res('late'), budget); });
    const r = await Promise.race([y.scoresAt(trig.time).catch(() => null), late]);
    clearTimeout(timer);

    if (r === 'late') return this._pass(trig, 'timeout');
    if (!r) return this._pass(trig, 'unseen');
    const d = SB.speechDecide(r, this.theta);
    this._note(trig, d);
    if (d.reject) { this.stats.기각++; return 'reject'; }
    this.stats.통과++;
    this.emit(trig);
    return 'keep';
  }

  _pass(trig, why) {
    if (why === 'timeout') this.stats.시간초과_통과++;
    else this.stats.판정없음_통과++;
    this._note(trig, null, why);
    this.emit(trig);
    return why;
  }

  _note(trig, d, why) {
    this.recent.push({
      onset: +trig.time.toFixed(2),
      판정: d ? (d.reject ? '기각' : '통과') : why,
      ...(d ? { Speech: +d.speech.toFixed(3), 무서운소리최대: +d.scare.toFixed(3) } : {}),
    });
    if (this.recent.length > 20) this.recent.shift();
  }

  report() { return { θ: this.theta, ...this.stats, 최근: this.recent }; }
};
