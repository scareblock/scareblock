/**
 * 고정 크기 표본 수집기.
 *
 * 렌더 루프 안에서 쓰므로 **재생 중에 할당하지 않는다.** 미리 잡은
 * Float64Array를 링으로 돌려 쓰고, 정렬은 보고할 때만 한다.
 * 계측이 계측 대상을 바꾸는 함정을 두 번 겪었다 (poc/README.md 참고).
 */
window.SB = window.SB || {};

SB.Samples = class {
  constructor(capacity = 4096) {
    this.buf = new Float64Array(capacity);
    this.seen = 0;
  }

  push(v) {
    this.buf[this.seen % this.buf.length] = v;
    this.seen++;
  }

  get count() {
    return Math.min(this.seen, this.buf.length);
  }

  /**
   * 선형 보간 없는 최근접 순위 백분위수. p는 0~100.
   *
   * eval/evaluate.py 의 latency_summary 는 np.percentile(선형 보간)이라 정의가 다르다.
   * 최근접 순위는 실제 표본 중 하나를 고르고 보간값 이상이 되므로 더 보수적이다
   * — render_s 를 부풀리는 쪽이라 적시성 판정에는 안전한 방향이다(#42 리뷰 🟢).
   */
  percentile(p) {
    const n = this.count;
    return n ? SB.Samples._rank(this._sorted(), p) : null;
  }

  /** 보고할 때 한 번만 정렬해 세 값을 뽑는다. */
  summary() {
    const n = this.count;
    if (!n) return null;
    const sorted = this._sorted();
    const at = (p) => +SB.Samples._rank(sorted, p).toFixed(2);
    return { n, p50: at(50), p90: at(90), max: at(100) };
  }

  _sorted() {
    return Float64Array.prototype.slice.call(this.buf, 0, this.count).sort();
  }

  static _rank(sorted, p) {
    const n = sorted.length;
    return sorted[Math.min(n - 1, Math.max(0, Math.ceil(p / 100 * n) - 1))];
  }

  reset() {
    this.seen = 0;
  }
};
