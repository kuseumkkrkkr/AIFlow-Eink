"""실제 거부 trial의 경쟁 후보를 다음 회차의 기존 16개 대체 슬롯 안에서 보호한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rival_guard_v7 as previous

engine, guard = previous.engine, previous.guard
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_observed_rival_v10"
OBSERVED_ROLE = "observed_rejected_trial_rival"


def bank_records(bank):
    """동일 row/rank/rival 최대 deficit 기록을 재현 가능한 순서로 반환한다."""
    return sorted(bank.values(), key=lambda i: (-i["deficit"], i["row"], i["rank"], i["rival"]))


def make_selector(state, records):
    """v6 정상 slack·위험 prefix는 그대로 두고 실제 경쟁 후보와 인접 후보가 16개를 공유한다."""
    prior_selector = previous.previous.make_selector()
    def select(logits, y, spec, audit, bank):
        """현재 정상이며 고정 pair floor도 통과한 실제 후보만 여유 순으로 먼저 보호한다."""
        witnessed = state.get("bank", {})
        state["bank"] = {}
        prior = prior_selector(logits, y, spec, audit, bank)
        prefix = [i for i in prior if i["role"] in previous.PRIORITY_ROLES]
        chosen, seen = [], set()
        def add(item):
            """원 floor·기존 위험 prefix를 유지하며 중복과 전체 64 방향 상한을 검사한다."""
            key = (item["row"], item["rank"], item["rival"])
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item)
                return True
            return False
        for item in prefix:
            add(item)
        observed = []
        for item in witnessed.values():
            row, k, rival = item["row"], item["rank"], item["rival"]
            floor = float(spec[f"top{k}_floor"][row])
            margin = float(logits[row, int(y[row])]-logits[row, rival])
            if audit["safe"][k][row] and margin >= floor and (row, k, rival) not in seen:
                observed.append((margin-floor, row, k, rival))
        observed.sort()
        admitted = 0
        for _, row, k, rival in observed:
            if admitted == previous.ALTERNATIVE_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
            if add(dict(row=row, rank=k, rival=rival, floor=float(spec[f"top{k}_floor"][row]), role=OBSERVED_ROLE)):
                admitted += 1
        observed_admitted = admitted
        order = np.argsort(-logits, axis=1, kind="stable"); alternatives = []
        for item in prefix:
            row, k = item["row"], item["rank"]
            others = order[row][order[row] != y[row]]; rival = int(others[1 if k == 1 else 5])
            gap = float(logits[row, item["rival"]]-logits[row, rival]); assert gap >= 0
            alternatives.append((gap, row, k, rival))
        for _, row, k, rival in sorted(alternatives):
            if admitted == previous.ALTERNATIVE_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
            if add(dict(row=row, rank=k, rival=rival, floor=float(spec[f"top{k}_floor"][row]), role="priority_near_alternative_rival")):
                admitted += 1
        for item in prior:
            if item["role"] not in previous.PRIORITY_ROLES:
                add(item)
        record = dict(round=len(records), eligible_observed_candidates=[dict(row=row, rank=k, rival=rv, fixed_slack=slack)
            for slack, row, k, rv in observed], observed_admitted=observed_admitted, combined_alternative_admitted=admitted,
            next_exact_interference_bank=[])
        records.append(record); state["record"] = record
        return chosen
    return select


def capture_rejected(state, allowed, newly_bad):
    """새 모델 호출 없이 거부된 실제 rank 검사에서 나온 모든 경쟁 key의 최대 deficit를 기억한다."""
    if not allowed and "record" in state:
        for item in newly_bad:
            key = (item["row"], item["rank"], item["rival"])
            if key not in state["bank"] or item["deficit"] > state["bank"][key]["deficit"]:
                state["bank"][key] = dict(item)
        state["record"]["next_exact_interference_bank"] = bank_records(state["bank"])


def run() -> int:
    """원 parent에서 시작하여 실제 경쟁 기록 외에는 봉인 v7 예산·점수·승격 규칙을 바꾸지 않는다."""
    plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    original_output, original_factory, original_write = previous.OUTPUT, previous.make_selector, engine._write
    original_accept = guard.accept_guard
    state, records = {}, []
    def accept_logged(before, after):
        """기존 accept 반환을 그대로 두고 거부된 실제 경쟁 key만 수동 기록한다."""
        allowed, reason, newly_bad = original_accept(before, after)
        capture_rejected(state, allowed, newly_bad)
        return allowed, reason, newly_bad
    def write_variant(path, data):
        """v7 기록 hook 뒤에서 관측 후보 선택 계약과 회차별 증거를 새 출력에 봉인한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_file=Path(__file__).name, entrypoint_sha256=engine._sha(Path(__file__)),
                observed_rival_priority_revision="v10", observed_rival_cap=previous.ALTERNATIVE_CAP,
                predecessor_result_sha256=verified["result_sha256"],
                observed_memory="all exact row/rank/rival keys causing original guard rejection over immediately prior round only; maximum deficit per key; initial empty",
                selection="unchanged v7 risk/safe primary prefix; currently-safe observed rival pairs passing their immutable floor, sorted by current fixed-pair slack, share the original 16 slots with second/sixth OTHER rivals; v6 filler up to 64",
                changed_factors="only alternative-slot use of actual rejected-trial competitors versus v7; original floors, primary safe slack priority, rank guards, alpha, solver, 64/512 direction caps and forwards unchanged")
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
        if path == OUTPUT / "active_trust_result.json":
            assert len(records) == len(data["history"])
            data["observed_rival_selection_records"] = records
        original_write(path, data)
    previous.OUTPUT, previous.make_selector = OUTPUT, lambda: make_selector(state, records)
    guard.accept_guard, engine._write = accept_logged, write_variant
    try:
        return previous.run()
    finally:
        previous.OUTPUT, previous.make_selector, engine._write = original_output, original_factory, original_write
        guard.accept_guard = original_accept


def self_test() -> int:
    """Top-1 세 번째·Top-5 일곱 번째 실제 경쟁 후보, 강한 pair 제외, 빈 시작 parity와 16/64 상한을 검사한다."""
    y = np.zeros(2048, dtype=np.int64)
    z = np.full((2048, 372), -100., dtype=np.float32); z[:, 1] = 0.; z[:, 0] = 2.
    z[:4, 0] = [.1, .2, .3, .4]; z[40, :4] = [1.01, 0., -.001, -.02]
    z[1040, :8] = [.5, 2., 1.5, 1., .8, .4, .399, .398]
    spec = {"top1_mask": np.ones(2048, dtype=bool), "top1_floor": np.ones(2048, dtype=np.float32),
            "top5_mask": np.zeros(2048, dtype=bool), "top5_floor": np.zeros(2048, dtype=np.float32)}
    spec["top1_mask"][1040] = False; spec["top5_mask"][1040] = True; spec["top5_floor"][1040] = .09
    bank = [dict(row=40, rank=1, deficit=.001), dict(row=1040, rank=5, deficit=.001)]
    audit = guard.inspect_guard(z, y, spec)
    assert make_selector({}, [])(z, y, spec, audit, bank) == previous.make_selector()(z, y, spec, audit, bank)
    witnesses = [dict(row=row, rank=k, rival=rv, floor=float(spec[f"top{k}_floor"][row]), margin=-.2, deficit=1.)
                 for row, k, rv in ((40, 1, 3), (1040, 5, 7), (1040, 5, 1))]
    state = {"bank": {tuple(i[n] for n in ("row", "rank", "rival")): i for i in witnesses}}
    records = []; chosen = make_selector(state, records)(z, y, spec, audit, bank)
    observed = [i for i in chosen if i["role"] == OBSERVED_ROLE]
    assert {(i["row"], i["rank"], i["rival"]) for i in observed} == {(40, 1, 3), (1040, 5, 7)}
    assert len(chosen) <= 64 and len({(i["row"], i["rank"], i["rival"]) for i in chosen}) == len(chosen)
    assert sum(i["role"] in (OBSERVED_ROLE, "priority_near_alternative_rival") for i in chosen) <= 16
    assert all(i["floor"] == float(spec[f'top{i["rank"]}_floor'][i["row"]]) for i in chosen)
    capture_rejected(state, True, witnesses); assert not state["bank"]
    capture_rejected(state, False, witnesses); assert len(state["bank"]) == 3
    duplicate = dict(witnesses[0], deficit=.1); capture_rejected(state, False, [duplicate])
    assert state["bank"][(40, 1, 3)]["deficit"] == 1.
    print(json.dumps(dict(self_test="pass", observed_nonadjacent_top1_top5_rivals=True,
        already_failed_strong_pair_excluded=True, empty_bank_v7_selection_exact=True,
        rejected_only_max_deficit_capture=True, fixed_floors_and_16_64_quotas_preserved=True)))
    return 0


def main() -> int:
    """별도 봉인 연구 출력 실행 또는 모델을 호출하지 않는 경쟁 후보 검증에 진입한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
