"""직전 trial의 실제 rank margin 낙폭/현재 slack으로 정상 보호 슬롯 순위만 비교한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rival_guard_v7 as previous
import run_hwr_active_trust_rank_persistence_v4 as rank_history

engine = previous.engine
guard = previous.guard
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_response_priority_v9"
DENOMINATOR_MIN = 1e-4
RANK_ROLES = {"retained_rank_risk_until_fixed_floor_pass", "currently_correct_near_rank_boundary"}


def response_key(row, k, rank, spec, response):
    """동일 TRAIN의 직전 실제 rank 낙폭을 현재 원 floor slack으로 나누며 확률로 해석하지 않는다."""
    slack = float(rank[f"top{k}_margin"][row]-spec[f"top{k}_floor"][row])
    drop = 0. if response is None else float(response[k][row])
    return (-drop/max(DENOMINATOR_MIN, slack), slack, row, k)


def make_selector(state, records):
    """v6 기억·v4 위험 유지·v7 대체 rival은 그대로 두고 16개 정상 슬롯 순위만 바꾼다."""
    rank_selector = rank_history.make_selector(); remembered = set()
    def select(logits, y, spec, audit, bank):
        """직전 낙폭을 동결한 뒤 새 회차 수집을 시작하고 같은 64개 방향 상한 안에서 선택한다."""
        response = state.get("drops")
        state["before"] = {k: audit["rank"][f"top{k}_margin"].copy() for k in (1, 5)}
        state["drops"] = {k: np.zeros_like(state["before"][k]) for k in (1, 5)}
        def keep(row, k):
            """선택 기억 해제는 원래 v6의 현재 정상 여부·0.25 floor buffer 규칙을 유지한다."""
            slack = float(audit["rank"][f"top{k}_margin"][row]-spec[f"top{k}_floor"][row])
            return bool(audit["safe"][k][row] and slack < max(engine.SLACK, previous.previous.SAFE_BUFFER_FRACTION*abs(float(spec[f"top{k}_floor"][row]))))
        remembered.intersection_update({key for key in remembered if keep(*key)})
        fresh = {(i["row"], i["rank"]) for i in bank if audit["safe"][i["rank"]][i["row"]]}
        remembered.update(key for key in fresh if keep(*key))
        assert len(remembered) <= 2*len(y)
        prior = rank_selector(logits, y, spec, audit, bank)
        chosen = []; seen = set()
        def add(item):
            """현재 rival·원 floor·원 위험 prefix를 보존하고 중복 및 전체 방향 상한을 지킨다."""
            key = (item["row"], item["rank"], item["rival"])
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item)
                return True
            return False
        for item in prior:
            if item["role"] in RANK_ROLES:
                add(item)
        candidates = sorted(response_key(row, k, audit["rank"], spec, response) for row, k in fresh | remembered)
        admitted = 0
        for _, _, row, k in candidates:
            role = "rejected_currently_safe_near_floor" if (row, k) in fresh else "remembered_safe_interference_until_buffer"
            if add(dict(row=row, rank=k, rival=int(audit["rank"][f"top{k}_rival"][row]), floor=float(spec[f"top{k}_floor"][row]), role=role)):
                admitted += 1
            if admitted == previous.previous.previous.SAFE_BANK_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
        records.append(dict(round=len(records), eligible_safe_candidates=len(candidates), admitted_safe_candidates=admitted,
            candidate_order=[dict(row=row, rank=k, response_ratio=-score, slack=slack) for score, slack, row, k in candidates]))
        order = np.argsort(-logits, axis=1, kind="stable"); alternatives = []
        for item in chosen.copy():
            row, k = item["row"], item["rank"]
            others = order[row][order[row] != y[row]]; rival = int(others[1 if k == 1 else 5])
            gap = float(logits[row, item["rival"]]-logits[row, rival]); assert gap >= 0
            alternatives.append((gap, row, k, rival))
        admitted = 0
        for _, row, k, rival in sorted(alternatives):
            if add(dict(row=row, rank=k, rival=rival, floor=float(spec[f"top{k}_floor"][row]), role="priority_near_alternative_rival")):
                admitted += 1
            if admitted == previous.ALTERNATIVE_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
        for item in prior:
            if item["role"] not in RANK_ROLES:
                add(item)
        return chosen
    return select


def run() -> int:
    """봉인 v7 driver를 재사용해 제안 여유는 원래 1e-4로 두며 새로운 실험 출력만 만든다."""
    plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verification = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verification["result_sha256"]
    original_output, original_factory, original_write = previous.OUTPUT, previous.make_selector, engine._write
    original_inspect = guard.inspect_guard
    state = {}; records = []
    def inspect_response(logits, y, spec):
        """기존 실제 rank 검사 반환은 유지하며 같은 호출에서만 회차 내 최대 실제 margin 낙폭을 기록한다."""
        audit = original_inspect(logits, y, spec)
        if "before" in state:
            for k in (1, 5):
                state["drops"][k] = np.maximum(state["drops"][k], np.maximum(state["before"][k]-audit["rank"][f"top{k}_margin"], 0.))
        return audit
    def write_variant(path, data):
        """v7 기록 hook 뒤에서 최초 계획 및 순위 증거를 봉인하고 기존 소스·결과는 덮지 않는다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_file=Path(__file__).name, entrypoint_sha256=engine._sha(Path(__file__)),
                safe_response_priority_revision="v9", response_denominator_min=DENOMINATOR_MIN,
                response_memory="maximum positive BEFORE actual rank margin minus trial actual rank margin over immediately prior round, including rival switches; cold start empty",
                predecessor_result_sha256=verification["result_sha256"],
                selection="v7 with only safe-slot ordering changed to descending prior observed drop / max(1e-4, CURRENT slack); ties by slack,row,rank; unchanged 16/64 quotas",
                changed_factors="only relative observed-response ranking of safe slots versus v7; no v8 buffers, new gradients, forwards, acceptance floor changes or architecture changes")
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
        if path == OUTPUT / "active_trust_result.json":
            assert len(records) == len(data["history"])
            data["safe_response_selection_records"] = records
        original_write(path, data)
    previous.OUTPUT, previous.make_selector = OUTPUT, lambda: make_selector(state, records)
    guard.inspect_guard, engine._write = inspect_response, write_variant
    try:
        return previous.run()
    finally:
        previous.OUTPUT, previous.make_selector, engine._write = original_output, original_factory, original_write
        guard.inspect_guard = original_inspect


def self_test() -> int:
    """작은 raw slack만 우선하는 반례·실제 낙폭 비율·빈 시작·변경 없는 원 floor를 검사한다."""
    rank = {"top1_margin": np.array([.11, .2], dtype=np.float32)}
    spec = {"top1_floor": np.array([.1, .1], dtype=np.float32)}
    response = {1: np.array([.001, .3], dtype=np.float32)}
    assert sorted([response_key(i, 1, rank, spec, None) for i in (0, 1)])[0][2] == 0
    assert sorted([response_key(i, 1, rank, spec, response) for i in (0, 1)])[0][2] == 1
    assert np.array_equal(spec["top1_floor"], np.array([.1, .1], dtype=np.float32))
    print(json.dumps(dict(self_test="pass", greater_observed_relative_risk_prioritized=True, cold_start_preserves_slack_order=True, original_floors_unchanged=True)))
    return 0


def main() -> int:
    """새 비교 실험 또는 숫자만으로 구성한 선택 우선순위 반례 검사에 진입한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
