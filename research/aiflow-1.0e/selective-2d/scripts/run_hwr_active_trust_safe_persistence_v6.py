"""한 번 발견한 정상 간섭 조건을 고정 floor 대비 사전 선언 여유까지 기억한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_safe_priority_v5 as previous

engine = previous.engine
guard = previous.guard
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_safe_persistence_v6"
SAFE_BUFFER_FRACTION = .25


def make_selector():
    """직전 실제 거부 조건 중 현재 정상이며 buffer 미회복인 row/rank만 세션 내 기억한다."""
    rank_selector = previous.previous.make_selector()
    remembered = set()
    def select(logits, y, spec, audit, bank):
        """v4 위험 prefix를 보존하고 기억+직전 정상 간섭을 같은 16개 quota·64개 상한으로 선택한다."""
        def needs_memory(row, k):
            """선택 우선순위 buffer만 사용하며 실제 승패는 원래 불변 floor로 판정한다."""
            slack = float(audit["rank"][f"top{k}_margin"][row]-spec[f"top{k}_floor"][row])
            buffer = max(engine.SLACK, SAFE_BUFFER_FRACTION*abs(float(spec[f"top{k}_floor"][row])))
            return bool(audit["safe"][k][row] and slack < buffer)
        remembered.intersection_update({(row, k) for row, k in remembered if needs_memory(row, k)})
        fresh = {(int(item["row"]), int(item["rank"])) for item in bank
                 if audit["safe"][int(item["rank"])][int(item["row"])]}
        remembered.update((row, k) for row, k in fresh if needs_memory(row, k))
        assert len(remembered) <= 2*len(y)
        prior = rank_selector(logits, y, spec, audit, bank)
        chosen = []; seen = set()
        def add(item):
            """현재 rival·원래 floor로 중복을 제거해 총 미분 방향을 제한한다."""
            key = (item["row"], item["rank"], item["rival"])
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item)
                return True
            return False
        for item in prior:
            if item["role"] in previous.RISK_ROLES:
                add(item)
        admitted = 0
        for _, row, k in sorted((float(audit["rank"][f"top{k}_margin"][row]-spec[f"top{k}_floor"][row]), row, k)
                               for row, k in fresh | remembered):
            role = "rejected_currently_safe_near_floor" if (row, k) in fresh else "remembered_safe_interference_until_buffer"
            if add(dict(row=row, rank=k, rival=int(audit["rank"][f"top{k}_rival"][row]),
                        floor=float(spec[f"top{k}_floor"][row]), role=role)):
                admitted += 1
            if admitted == previous.SAFE_BANK_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
        for item in prior:
            if item["role"] not in previous.RISK_ROLES:
                add(item)
        return chosen
    return select


def run() -> int:
    """v5와 같은 원 parent·입력·solver·512 방향 예산에서 정상 간섭 기억만 비교한다."""
    plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    for name, digest in plan["dependencies"].items():
        assert engine._sha(engine.ROOT / "scripts" / name) == digest
    original_inspect, original_accept, original_select, original_write = engine.inspect, engine.accept, engine.select, engine._write
    def write_variant(path, data):
        """최초 실행 전에 buffer 가설을 봉인하며 제품 gate·원 margin floor는 변경하지 않는다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)), entrypoint_file=Path(__file__).name,
                rank_guard_revision="v2", rank_persistence_revision="v4", safe_interference_priority_revision="v5",
                safe_interference_persistence_revision="v6", safe_memory_buffer_fraction=SAFE_BUFFER_FRACTION,
                safe_memory_buffer_min=engine.SLACK, rejected_safe_bank_cap=previous.SAFE_BANK_CAP,
                predecessor_result_sha256=verified["result_sha256"],
                selection="unchanged v4 rank prefix; currently-safe fresh plus previously discovered safe constraints until current slack >= max(1e-4, 0.25*abs(original floor)); same current-slack ordering and 16 quota; v4 fill to 64",
                actual_step_acceptance=plan["actual_step_acceptance"],
                changed_factors="only historical persistence of discovered currently-safe bank conditions versus v5; buffer is a predeclared selection hypothesis, not a human probability or changed acceptance floor")
            data["dependencies"].update(plan["dependencies"])
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
        original_write(path, data)
    engine.inspect, engine.accept, engine.select, engine._write = guard.inspect_guard, guard.accept_guard, make_selector(), write_variant
    try:
        return engine.run(OUTPUT)
    finally:
        engine.inspect, engine.accept, engine.select, engine._write = original_inspect, original_accept, original_select, original_write


def self_test() -> int:
    """bank에서 사라진 임의 정상 행 유지·buffer 회복 시 해제·세션 격리·불변 floor를 검사한다."""
    y = np.zeros(2048, dtype=np.int64)
    z = np.full((2048, 372), -100., dtype=np.float32); z[:, 1] = 0.; z[:, 0] = 2.
    z[:4, 0] = [.1, .2, .3, .4]; z[40, 0] = 1.01
    spec = {"top1_mask": np.ones(2048, dtype=bool), "top1_floor": np.ones(2048, dtype=np.float32),
            "top5_mask": np.zeros(2048, dtype=bool), "top5_floor": np.zeros(2048, dtype=np.float32)}
    selector = make_selector(); bank = [dict(row=40, rank=1, deficit=.001)]
    first = selector(z, y, spec, guard.inspect_guard(z, y, spec), bank)
    second = selector(z, y, spec, guard.inspect_guard(z, y, spec), [])
    assert any(i["row"] == 40 and i["role"] == "remembered_safe_interference_until_buffer" for i in second)
    # 신규 세션에는 과거 실험의 실패/기억을 자동 주입하지 않는다.
    fresh = make_selector()(z, y, spec, guard.inspect_guard(z, y, spec), [])
    assert not any(i["role"] == "remembered_safe_interference_until_buffer" for i in fresh)
    recovered = z.copy(); recovered[40, 0] = 1.25
    third = selector(recovered, y, spec, guard.inspect_guard(recovered, y, spec), [])
    assert not any(i["row"] == 40 and i["role"] == "remembered_safe_interference_until_buffer" for i in third)
    for items in (first, second, third):
        assert len(items) <= 64 and len({(i["row"], i["rank"], i["rival"]) for i in items}) == len(items)
        assert all(i["floor"] == float(spec[f'top{i["rank"]}_floor'][i["row"]]) for i in items)
    print(json.dumps(dict(self_test="pass", disappearing_bank_risk_remembered=True,
        buffer_recovery_releases_memory=True, session_memory_isolated=True, original_floors_unchanged=True)))
    return 0


def main() -> int:
    """새로운 독립 실험 또는 모델 갱신 없는 결정 규칙 단위 검증만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
