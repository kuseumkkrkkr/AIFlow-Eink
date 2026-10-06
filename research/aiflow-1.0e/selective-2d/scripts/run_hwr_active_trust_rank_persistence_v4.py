"""현재 맞는 위험 행을 고정 floor 회복까지 유지하는 TRAIN active-set 실험."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rank_priority_v3 as previous

engine = previous.engine
guard = previous.guard
BASE_SELECT = previous.BASE_SELECT
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_rank_persistence_v4"


def make_selector():
    """세션별 위험 row/rank만 보관하며 실제 현재 rival은 매 회차 새로 구한다."""
    pending = set()
    def select(logits, y, spec, audit, bank):
        """floor를 회복하지 못한 기존 위험 행을 먼저 유지하고 새 경계 4개/도메인을 추가한다."""
        pending.intersection_update({(row, k) for row, k in pending
            if audit["rank"][f"top{k}_mask"][row] and audit["rank"][f"top{k}_margin"][row] < spec[f"top{k}_floor"][row]})
        chosen = []; seen = set()
        def add(row, k, role):
            """현재 rival·원래 고정 floor로 중복 없이 64개 이하의 제약을 만든다."""
            rival = int(audit["rank"][f"top{k}_rival"][row]); key = (row, k, rival)
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(dict(row=row, rank=k, rival=rival,
                    floor=float(spec[f"top{k}_floor"][row]), role=role))
                if audit["rank"][f"top{k}_margin"][row] < spec[f"top{k}_floor"][row]:
                    pending.add((row, k))
        for _, row, k in sorted((float(audit["rank"][f"top{k}_margin"][row]), row, k) for row, k in pending):
            add(row, k, "retained_rank_risk_until_fixed_floor_pass")
        for lower, upper in ((0, 1024), (1024, len(y))):
            risk = [(float(audit["rank"][f"top{k}_margin"][row]), row, k)
                    for k in (1, 5) for row in range(lower, upper)
                    if spec[f"top{k}_mask"][row] and audit["rank"][f"top{k}_mask"][row]]
            for _, row, k in sorted(risk)[:4]:
                add(row, k, "currently_correct_near_rank_boundary")
        for item in BASE_SELECT(logits, y, spec, audit, bank):
            key = (item["row"], item["rank"], item["rival"])
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item)
        return chosen
    return select


def run() -> int:
    """v3 실패를 보존하며 선택 유지 여부만 바꾼 독립 입력/예산 동일 실험을 봉인한다."""
    prior_plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == prior_plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    original_inspect, original_accept, original_select, original_write = engine.inspect, engine.accept, engine.select, engine._write
    def write_variant(path, data):
        """처음 plan 저장 전에 상태 유지 규칙·실행 진입점·모든 불변 조건을 명시한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)), entrypoint_file=Path(__file__).name,
                rank_guard_revision="v2", rank_persistence_revision="v4",
                predecessor_result_sha256=verified["result_sha256"],
                selection="retain previously selected rank risks until actual fixed floor passes; then four current risks/domain and v1 fill to 64",
                actual_step_acceptance=prior_plan["actual_step_acceptance"],
                changed_factors="only persistence of selected near-rank risks versus v3; unchanged solver, floors, rank guard and direction caps")
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
            data["dependencies"][Path(guard.__file__).name] = engine._sha(Path(guard.__file__))
        original_write(path, data)
    engine.inspect, engine.accept, engine.select, engine._write = guard.inspect_guard, guard.accept_guard, make_selector(), write_variant
    try:
        return engine.run(OUTPUT)
    finally:
        engine.inspect, engine.accept, engine.select, engine._write = original_inspect, original_accept, original_select, original_write


def self_test() -> int:
    """첫 회차 위험 반례가 다음 회차 quota 밖이어도 원 floor 미회복이면 유지되는지 검사한다."""
    load = lambda name: np.load(engine.REFERENCE / f"{name}.npy", allow_pickle=False)
    y = load("joint_labels"); spec = {f"top{k}_{name}": load(f"reference_top{k}_{name}") for k in (1, 5) for name in ("mask", "floor")}
    z0 = np.concatenate((load("parent_old_logits"), load("parent_source_logits")))
    z1 = np.load(previous.OUTPUT / "first_trial_logits_03.npy", allow_pickle=False)
    selector = make_selector()
    first = selector(z0, y, spec, guard.inspect_guard(z0, y, spec), [])
    second = selector(z1, y, spec, guard.inspect_guard(z1, y, spec), [])
    assert any(item["row"] == 852 for item in first)
    assert any(item["row"] == 852 and item["role"] == "retained_rank_risk_until_fixed_floor_pass" for item in second)
    for items in (first, second):
        assert len(items) <= 64 and len({(i["row"], i["rank"], i["rival"]) for i in items}) == len(items)
        assert all(i["floor"] == float(spec[f'top{i["rank"]}_floor'][i["row"]]) for i in items)
    print(json.dumps(dict(self_test="pass", risk_retained_after_quota_exit=True, floors_unchanged=True,
        first_directions=len(first), second_directions=len(second))))
    return 0


def main() -> int:
    """별도 유지 실험 또는 시간 순서 반례 검사만 실행하며 기존 증거를 덮지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
