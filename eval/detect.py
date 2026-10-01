"""규칙 기반 점프 스케어 탐지기.

핵심은 **탐지 지연을 사건마다 기록**하는 것이다. E1이 그 값으로
lookahead L을 쓸어가며 "도달 전 개입이 되는가"를 판정한다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .features import Features


@dataclass
class Params:
    quiet_of_peak: float = 0.15   # 직전 1초가 최근 최댓값 대비 이만큼 아래면 정적
    surge_of_quiet: float = 5.0   # 그 정적 대비 이만큼 뛰면 급등
    surge_of_peak: float = 0.30   # 최근 최댓값 대비 최소치 (잔물결 배제)
    drms_min: float = 0.0         # dRMS/dt 하한 — 0이면 기울기 조건을 끈다
    cooldown_s: float = 2.0
    silence_floor: float = 1e-4   # 무음 구간 판정 건너뛰기
    backtrack_max_s: float = 1.0  # 되짚기 상한 — 없으면 t=0까지 내려간다
    verify_s: float = 0.0         # 사후 검증 관찰 구간(초). 0이면 검증하지 않는다
    verify_level: float = 0.5     # 관찰이 끝날 때 탐지 시점의 이 배 이상이면 「아직 크다」 → 기각
    verify_tail_s: float = 0.1    # 관찰 구간 **끝** 이만큼의 평균으로 판정한다


@dataclass
class Detection:
    onset: float           # 자극이 올라가기 시작한 추정 시각
    fire: float            # 규칙이 확신에 이른 시각 (분석 창의 끝 + 관찰 구간)
    score: float
    clip_id: str = ""      # 클립 간 교차 매칭을 막는다 (#19 리뷰 🔴2)
    onset_capped: bool = False   # 되짚기가 상한에 걸렸다 = onset 추정 실패
    verify_capped: bool = False  # 클립이 먼저 끝나 관찰 구간을 다 못 봤다 = 검증 못 함
    source: str = "rule"   # 어느 경로가 낸 탐지인가 — §5 계약 배열의 source와 같은 뜻.
    # 한 표에 두 경로가 섞일 때 「되짚기 상한 N건」 같은 집계가 어느 쪽 것인지 갈라야 한다 (#43 리뷰 🟡2)

    @property
    def backtrack_s(self) -> float:
        """되짚은 거리. **적시성 지표가 아니다** — 적시성은 라벨 onset을 쓴다.

        #19 리뷰 🔴1: 이 값을 적시성에 쓰면 탐지기를 자기 자신과 비교하게 된다.
        진단용(되짚기가 어디서 멈추는지)으로만 본다.
        """
        return self.fire - self.onset


def _backtrack_onset(f: Features, i: int, max_s: float) -> tuple[float, bool]:
    """RMS가 정적 수준을 벗어나기 시작한 프레임까지 되짚는다.

    발화 시점이 아니라 시작점을 추정해야 "어디서 올라가기 시작했는가"를 볼 수 있다
    (labeling-guide §3 — onset은 정점이 아니라 시작점).

    **상한을 둔다.** 바닥이 조용해지지 않는 구간(계속 시끄러운 장면, 음악 위의
    비명)에서는 `j = 0`까지 내려가 추정이 무의미해진다 (#19 리뷰 🟡3).
    상한에 걸렸으면 그 사실을 함께 돌려주어 리포트에서 빈도를 센다.
    """
    floor = max(f.quiet[i] * 1.5, f.peak[i] * 0.02)
    limit = max(0, i - max(1, int(round(max_s / f.hop))))
    j = i
    while j > limit and f.rms[j] > floor:
        j -= 1
    return f.t(j), j == limit and f.rms[j] > floor


def _sustained(f: Features, i: int, p: Params) -> tuple[bool, bool]:
    """탐지 직후 관찰 구간에서 소리가 **이어지는가**. (이어짐, 구간이 잘렸는가)를 돌려준다.

    README 02단계 — 「소리가 이어지면 음악, 끊기면 점프 스케어다. 기존 온라인 탐지는 그 순간에
    판정해야 해 불가능한 경로다.」 지연 버퍼가 있어 **사건 이후**를 보고 거를 수 있는 것이
    이 프로젝트의 구조적 이점이고, 이 함수가 그 판정이다.

    판정은 **관찰 구간 전체의 평균이 아니라 끝부분(`verify_tail_s`)의 평균**으로 한다. 전체로 재면
    구간이 길수록 뒤쪽 조용한 부분이 평균을 끌어내려, **길게 볼수록 관대해지는** 거꾸로 된 기준이
    된다(44클립에서 0.2초 34% → 1초 63% 통과로 실제로 그렇게 나왔다). 끝을 보면 「그때도 아직
    큰가」를 묻게 되어, 짧은 관찰은 아직 감쇠 중인 진짜 사건까지 버리고 긴 관찰일수록 정확해진다 —
    이것이 「관찰을 늘리면 판정이 좋아지지만 개입이 늦는다」는 예산 문제의 실제 모양이다.

    **클립이 먼저 끝나면 기각하지 않는다.** 관찰할 시간이 없던 것과 관찰해서 통과한 것은
    다르지만, 못 본 것을 기각으로 세면 클립 끝의 사건이 조용히 사라진다 — 클립 경계에서
    참 사건을 잃는 것은 이 하니스가 #19·#26에서 이미 한 번씩 막아온 종류의 사고다.
    대신 잘렸다는 사실을 함께 돌려 리포트에서 센다.
    """
    w = max(1, int(round(p.verify_s / f.hop)))
    end = i + w
    capped = end >= len(f)
    tail = max(1, int(round(p.verify_tail_s / f.hop)))
    seg = f.rms[max(i + 1, end - tail + 1):min(end, len(f) - 1) + 1]
    if seg.size == 0:
        return False, True
    return (float(seg.mean()) >= f.rms[i] * p.verify_level and not capped), capped


def run(f: Features, p: Params | None = None, clip_id: str = "") -> list[Detection]:
    p = p or Params()
    out: list[Detection] = []
    cooldown_frames = int(round(p.cooldown_s / f.hop))
    blocked_until = -1

    for i in range(len(f)):
        if i < blocked_until or f.peak[i] < p.silence_floor:
            continue
        quiet, cur, peak = f.quiet[i], f.rms[i], f.peak[i]
        if not (quiet < peak * p.quiet_of_peak):
            continue
        if not (cur > max(quiet * p.surge_of_quiet, peak * p.surge_of_peak)):
            continue
        if p.drms_min > 0 and f.drms[i] < p.drms_min:
            continue

        onset, capped = _backtrack_onset(f, i, p.backtrack_max_s)

        # 사후 검증 — 관찰 구간만큼 더 보고 「이어지는 소리」면 버린다.
        # 확신 시각은 그만큼 뒤로 밀린다. 「탐지 시점 + 관찰 구간 ≤ 3초」가 예산이므로
        # 관찰을 늘리면 개입이 늦는다 (README 02단계).
        v_capped = False
        fire = f.t_ready(i)
        if p.verify_s > 0:
            sustained, v_capped = _sustained(f, i, p)
            if sustained:
                blocked_until = i + cooldown_frames   # 기각해도 재발화는 막는다
                continue
            fire += p.verify_s

        out.append(Detection(onset=onset, fire=fire, score=float(cur / (peak + 1e-9)),
                             clip_id=clip_id, onset_capped=capped, verify_capped=v_capped))
        blocked_until = i + cooldown_frames

    return out


def detections_per_minute(dets: list[Detection], duration_s: float) -> float:
    """분당 발화 수. 라벨을 쓰지 않으므로 오탐률이 아니라 발화율이다.

    같은 축의 실측 — PoC 104초 관측(`poc/README.md` 「탐지 오탐 — 104초에 9건」 → 5.2건/분)과
    평가셋 212.7분 스윕(기본 임계값 6.37건/분, 2026.09.15 · 44클립 — `docs/labeling-guide.md` §4).
    「사건 빽빽 7.9건/분」은 출처가 없어 뺐다.
    """
    return len(dets) / (duration_s / 60.0) if duration_s > 0 else 0.0
