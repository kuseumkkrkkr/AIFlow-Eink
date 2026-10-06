"""별도 rank guard를 유지하고 현재 맞는 근접 순위 경계를 먼저 선택하는 TRAIN 실험."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rank_guard_v2 as guard

engine = guard.engine
BASE_SELECT = engine.select
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_rank_priority_v3"


def select_rank_priority(logits, y, spec, audit, bank):
    """각 TRAIN 도메인에서 현재 맞는 보호 순위 margin 최소 4개를 먼저 넣고 기존 후보로 채운다."""
    chosen = []; seen = set()
    for lower, upper in ((0, 1024), (1024, len(y))):
        risk = [(float(audit["rank"][f"top{k}_margin"][row]), row, k)
                for k in (1, 5) for row in range(lower, upper)
                if spec[f"top{k}_mask"][row] and audit["rank"][f"top{k}_mask"][row]]
        for _, row, k in sorted(risk)[:4]:
            rival = int(audit["rank"][f"top{k}_rival"][row]); seen.add((row, k, rival))
            chosen.append(dict(row=row, rank=k, rival=rival, floor=float(spec[f"top{k}_floor"][row]),
                               role="currently_correct_near_rank_boundary"))
    for item in BASE_SELECT(logits, y, spec, audit, bank):
        key = (item["row"], item["rank"], item["rival"])
        if key not in seen and len(chosen) < engine.DIRECTION_CAP:
            chosen.append(item); seen.add(key)
    return chosen


def run() -> int:
    """원 후보에서 출발해 선택 우선순위만 바꾸며 같은 rank/floor/예산 조건을 plan에 봉인한다."""
    previous_plan = json.loads((guard.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    previous_verification = json.loads((guard.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(guard.__file__)) == previous_plan["entrypoint_sha256"]
    assert engine._sha(Path(engine.__file__)) == previous_plan["script_sha256"]
    assert engine._sha(guard.OUTPUT / "active_trust_result.json") == previous_verification["result_sha256"]
    original_inspect, original_accept, original_write = engine.inspect, engine.accept, engine._write
    def write_variant(path, data):
        """실제 진입점·선택 규칙·그 외 불변 조건을 첫 plan 저장 전에 명시한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)), entrypoint_file=Path(__file__).name,
                rank_guard_revision="v2", rank_priority_revision="v3",
                predecessor_result_sha256=previous_verification["result_sha256"],
                selection="first four currently-correct protected rank boundaries per domain, then v1 selection with deduplication to 64",
                actual_step_acceptance=previous_plan["actual_step_acceptance"],
                changed_factors="only near-rank-boundary selection priority versus rank-guard v2; unchanged floors, rank guard, solver and budgets")
            data["dependencies"][Path(guard.__file__).name] = engine._sha(Path(guard.__file__))
        original_write(path, data)
    engine.inspect, engine.accept, engine.select, engine._write = guard.inspect_guard, guard.accept_guard, select_rank_priority, write_variant
    try:
        return engine.run(OUTPUT)
    finally:
        engine.inspect, engine.accept, engine.select, engine._write = original_inspect, original_accept, BASE_SELECT, original_write


def self_test() -> int:
    """봉인 TRAIN 반례의 누락을 복구하되 label/row 하드코딩 없이 예산·floor·중복을 검사한다."""
    load = lambda name: np.load(engine.REFERENCE / f"{name}.npy", allow_pickle=False)
    z = np.concatenate((load("parent_old_logits"), load("parent_source_logits")))
    y = load("joint_labels")
    spec = {f"top{k}_{name}": load(f"reference_top{k}_{name}") for k in (1, 5) for name in ("mask", "floor")}
    audit = guard.inspect_guard(z, y, spec)
    old = BASE_SELECT(z, y, spec, audit, [])
    new = select_rank_priority(z, y, spec, audit, [])
    assert not any(item["row"] == 852 for item in old)
    assert any(item["row"] == 852 and item["role"] == "currently_correct_near_rank_boundary" for item in new)
    assert len(new) <= 64 and len({(i["row"], i["rank"], i["rival"]) for i in new}) == len(new)
    for item in new:
        assert item["floor"] == float(spec[f'top{item["rank"]}_floor'][item["row"]])
    guard.self_test()
    print(json.dumps(dict(selection_self_test="pass", omitted_counterexample_now_selected=True,
        no_label_or_row_in_selection_rule=True, floors_unchanged=True, directions=len(new))))
    return 0


def main() -> int:
    """기존 결과를 보존하고 별도 priority 실험 또는 해당 단위 검사만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
