"""브라우저 규칙 탐지기(SB.RuleCore)가 eval/detect.py와 같은 탐지를 내는지 본다.

E1이 잰 탐지기와 데모에서 도는 탐지기가 같아야 E1의 숫자가 데모의 숫자다.
같은 wav를 양쪽에 넣고 발화 시각·onset을 프레임 단위로 대조한다.

    python3 extension/test/rule-parity.py _local/dataset/wav/*.wav

파이썬은 RMS를 float32로, JS는 float64로 계산한다. 임계값에 아주 가깝게 걸친 프레임은
양쪽이 갈릴 수 있으므로 **불일치 건수를 그대로 보고**하고, 전체의 1%를 넘으면 실패로 둔다.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval import detect, features  # noqa: E402

DUMP = Path(__file__).with_name("rule-dump.js")
TOL = 0.0105   # 한 프레임(10 ms) — 반올림 차이만 허용한다


def main(paths: list[str]) -> int:
    total = miss = 0
    for p in paths:
        x, sr = features.load_wav(p)
        f = features.extract(x, sr)
        py = detect.run(f, detect.Params(surge_of_quiet=20.0))
        js = json.loads(subprocess.check_output(["node", str(DUMP), p]))

        js_left = list(js)
        unmatched = 0
        for d in py:
            hit = next((j for j in js_left if abs(j["fire"] - d.fire) <= TOL
                        and abs(j["onset"] - d.onset) <= TOL), None)
            if hit is None:
                unmatched += 1
            else:
                js_left.remove(hit)
        unmatched += len(js_left)
        total += max(len(py), len(js))
        miss += unmatched
        print(f"{Path(p).name}: python {len(py)} · js {len(js)} · 불일치 {unmatched}")

    rate = miss / total if total else 0.0
    print(f"합계 {total}건 중 불일치 {miss} ({rate:.1%})")
    return 0 if rate <= 0.01 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
