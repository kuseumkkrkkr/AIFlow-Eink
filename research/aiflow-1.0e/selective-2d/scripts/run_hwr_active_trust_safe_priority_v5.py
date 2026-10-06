"""거부 trial에서 발견한 현재 정상 조건을 실제 남은 floor 여유 순으로 보호한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rank_persistence_v4 as previous

engine = previous.engine
guard = previous.guard
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_safe_priority_v5"
SAFE_BANK_CAP = 16
RISK_ROLES = {"retained_rank_risk_until_fixed_floor_pass", "currently_correct_near_rank_boundary"}


def make_selector():
    """v4 위험 유지 상태를 보존하고 과거 거부된 정상 조건의 현재 slack을 다시 계산한다."""
    prior_selector = previous.make_selector()
    def select(logits, y, spec, audit, bank):
        """기존 위험 prefix, 정상 간섭 16개, 기존 filler 순으로 고정 floor·64개 상한을 지킨다."""
        prior = prior_selector(logits, y, spec, audit, bank)
        chosen = []; seen = set()
        def add(item):
            """row/rank/rival을 중복 제거하며 기존 상태의 위험 행을 먼저 보존한다."""
            key = (item["row"], item["rank"], item["rival"])
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item)
                return True
            return False
        for item in prior:
            if item["role"] in RISK_ROLES:
                add(item)
        safe_bank = {(int(item["row"]), int(item["rank"])) for item in bank
                     if audit["safe"][int(item["rank"])][int(item["row"])]}
        admitted = 0
        for slack, row, k in sorted((float(audit["rank"][f"top{k}_margin"][row]-spec[f"top{k}_floor"][row]), row, k)
                                   for row, k in safe_bank):
            item = dict(row=row, rank=k, rival=int(audit["rank"][f"top{k}_rival"][row]),
                        floor=float(spec[f"top{k}_floor"][row]), role="rejected_currently_safe_near_floor")
            if add(item):
                admitted += 1
            if admitted == SAFE_BANK_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
        for item in prior:
            if item["role"] not in RISK_ROLES:
                add(item)
        return chosen
    return select


def run() -> int:
    """같은 원 parent·2048 TRAIN·512 방향 예산에서 간섭 우선순위 한 요소만 바꾼다."""
    prior_plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == prior_plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    for name, digest in prior_plan["dependencies"].items():
        assert engine._sha(engine.ROOT / "scripts" / name) == digest
    original_inspect, original_accept, original_select, original_write = engine.inspect, engine.accept, engine.select, engine._write
    def write_variant(path, data):
        """최초 plan 저장 전에 비교 규칙·현재 slack 순서·입력/solver 불변 조건을 봉인한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)), entrypoint_file=Path(__file__).name,
                rank_guard_revision="v2", rank_persistence_revision="v4", safe_interference_priority_revision="v5",
                rejected_safe_bank_cap=SAFE_BANK_CAP, predecessor_result_sha256=verified["result_sha256"],
                selection="v4 rank-risk prefix; up to 16 currently-safe previous-trial constraints by CURRENT actual margin minus immutable floor; then v4 fill to 64",
                actual_step_acceptance=prior_plan["actual_step_acceptance"],
                changed_factors="only priority of actually rejected currently-safe bank constraints versus v4; unchanged solver, rank persistence, floors, alphas, rows and direction caps")
            data["dependencies"].update(prior_plan["dependencies"])
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
        original_write(path, data)
    engine.inspect, engine.accept, engine.select, engine._write = guard.inspect_guard, guard.accept_guard, make_selector(), write_variant
    try:
        return engine.run(OUTPUT)
    finally:
        engine.inspect, engine.accept, engine.select, engine._write = original_inspect, original_accept, original_select, original_write


def self_test() -> int:
    """과거 실제 거부 조건을 보편 규칙으로 재선택하고 label/row 하드코딩 없이 경계 순서를 검사한다."""
    load = lambda name: np.load(engine.REFERENCE / f"{name}.npy", allow_pickle=False)
    y = load("joint_labels"); spec = {f"top{k}_{name}": load(f"reference_top{k}_{name}") for k in (1, 5) for name in ("mask", "floor")}
    result = json.loads((previous.OUTPUT / "active_trust_result.json").read_text(encoding="utf-8"))
    z = np.load(previous.OUTPUT / "active_trust_logits.npy", allow_pickle=False)
    audit = guard.inspect_guard(z, y, spec)
    bank = result["history"][-1]["next_interference_bank"]
    selected = make_selector()(z, y, spec, audit, bank)
    assert any(item["row"] == 470 and item["rank"] == 1 and item["role"] == "rejected_currently_safe_near_floor" for item in selected)
    safe_selected = [item for item in selected if item["role"] == "rejected_currently_safe_near_floor"]
    slacks = [float(audit["rank"][f'top{i["rank"]}_margin'][i["row"]]-spec[f'top{i["rank"]}_floor'][i["row"]]) for i in safe_selected]
    assert slacks == sorted(slacks) and len(safe_selected) <= SAFE_BANK_CAP
    assert len(selected) <= 64 and len({(i["row"], i["rank"], i["rival"]) for i in selected}) == len(selected)
    assert all(i["floor"] == float(spec[f'top{i["rank"]}_floor'][i["row"]]) for i in selected)
    # 과거 거부는 정책 단위 검사에만 사용하며 새 실험의 첫 bank는 여전히 빈 목록이다.
    assert not any(i["role"] == "rejected_currently_safe_near_floor" for i in make_selector()(z, y, spec, audit, []))
    previous.self_test()
    print(json.dumps(dict(self_test="pass", prior_rejected_safe_case_selected=True,
        current_slack_order_verified=True, frozen_floors_unchanged=True, selected=len(selected), safe_bank_selected=len(safe_selected))))
    return 0


def main() -> int:
    """새 출력 run 또는 모델 갱신 없는 선택 정책 단위 검사만 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
