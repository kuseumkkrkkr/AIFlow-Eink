"""우선 보호 조건의 가까운 대체 rival을 같은 예산 안에서 먼저 선택하는 TRAIN 실험."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_safe_persistence_v6 as previous

engine = previous.engine
guard = previous.guard
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_rival_guard_v7"
ALTERNATIVE_CAP = 16
PRIORITY_ROLES = previous.previous.RISK_ROLES | {"rejected_currently_safe_near_floor", "remembered_safe_interference_until_buffer"}


def make_selector():
    """v6 상태를 그대로 유지하며 위험/정상 prefix의 가까운 second/sixth OTHER를 먼저 보호한다."""
    prior_selector = previous.make_selector()
    def select(logits, y, spec, audit, bank):
        """원 prefix·가까운 대체 16개·기존 filler 순서로 현재 고정 floor와 64개 상한을 지킨다."""
        prior = prior_selector(logits, y, spec, audit, bank)
        prefix = [item for item in prior if item["role"] in PRIORITY_ROLES]
        chosen = []; seen = set()
        def add(item):
            """행·rank·rival 중복을 제거하며 상태 유지에 필요한 원 prefix는 절대 밀어내지 않는다."""
            key = (item["row"], item["rank"], item["rival"])
            if key not in seen and len(chosen) < engine.DIRECTION_CAP:
                seen.add(key); chosen.append(item)
                return True
            return False
        for item in prefix:
            add(item)
        order = np.argsort(-logits, axis=1, kind="stable")
        alternatives = []
        for item in prefix:
            row, k = item["row"], item["rank"]
            others = order[row][order[row] != y[row]]
            rival = int(others[1 if k == 1 else 5])
            gap = float(logits[row, item["rival"]]-logits[row, rival])
            assert gap >= 0
            alternatives.append((gap, row, k, rival))
        admitted = 0
        for _, row, k, rival in sorted(alternatives):
            if add(dict(row=row, rank=k, rival=rival, floor=float(spec[f"top{k}_floor"][row]),
                        role="priority_near_alternative_rival")):
                admitted += 1
            if admitted == ALTERNATIVE_CAP or len(chosen) == engine.DIRECTION_CAP:
                break
        for item in prior:
            if item["role"] not in PRIORITY_ROLES:
                add(item)
        return chosen
    return select


def run() -> int:
    """원 parent·TRAIN·solver·512 방향 예산을 유지하고 추가 forward 없이 전체 trial 점수를 기록한다."""
    import run_hwr_affine_distillation_experiment_v1 as runtime
    plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    for name, digest in plan["dependencies"].items():
        assert engine._sha(engine.ROOT / "scripts" / name) == digest
    original_inspect, original_accept, original_select, original_write = engine.inspect, engine.accept, engine.select, engine._write
    original_predict = runtime._predict_logits
    trace = []; initial = None
    def predict_logged(model, x, device, batch):
        """실제 기존 predict 반환을 변경하지 않고 초기·각 trial·최종 372-way logits를 보존한다."""
        nonlocal initial
        actual = original_predict(model, x, device, batch)
        assert actual.shape == (2048, 372) and batch == 32 and np.isfinite(actual).all()
        index = len(trace); filename = f"full_logits_{index:03d}.npy"
        assert index < 2+engine.ROUNDS*len(engine.ALPHAS)
        trace.append(dict(file=filename))
        if not OUTPUT.exists():
            assert index == 0; initial = actual.copy()
        else:
            if initial is not None:
                np.save(OUTPUT / trace[0]["file"], initial, allow_pickle=False); initial = None
            np.save(OUTPUT / filename, actual, allow_pickle=False)
        return actual
    def write_variant(path, data):
        """선택·기록 계약을 처음 plan에 봉인하고 결과에 모든 forward의 hash/순서를 명시한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)), entrypoint_file=Path(__file__).name,
                rank_guard_revision="v2", rank_persistence_revision="v4", safe_interference_priority_revision="v5",
                safe_interference_persistence_revision="v6", safe_memory_buffer_fraction=previous.SAFE_BUFFER_FRACTION,
                safe_memory_buffer_min=engine.SLACK, rejected_safe_bank_cap=previous.previous.SAFE_BANK_CAP,
                priority_alternative_revision="v7", priority_alternative_cap=ALTERNATIVE_CAP,
                full_logit_trace_revision="v7", max_logged_forwards=2+engine.ROUNDS*len(engine.ALPHAS),
                extra_diagnostic_forwards=0, predecessor_result_sha256=verified["result_sha256"],
                selection="unchanged v6 primary prefix; up to 16 second/sixth OTHER rivals by current primary-minus-alternative logit gap; v6 filler up to 64",
                actual_step_acceptance=plan["actual_step_acceptance"],
                changed_factors="only earlier coverage of close alternative rivals for primary rank/safe conditions versus v6; unchanged original floors, guards, solver, alpha and direction budget; full logits are passive telemetry")
            data["dependencies"].update(plan["dependencies"])
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
        if path == OUTPUT / "active_trust_result.json":
            assert len(trace) == data["actual_model_checks"]+2
            for item in trace:
                item["sha256"] = engine._sha(OUTPUT / item["file"])
            trials = []; index = 1
            for history in data["history"]:
                for trial_id, _ in enumerate(history["trials"]):
                    trials.append(dict(round=history["round"], trial=trial_id, **trace[index])); index += 1
            assert index == len(trace)-1
            data["full_logits_trace"] = dict(initial=trace[0], trials=trials, final=trace[-1],
                recorded_forwards=len(trace), extra_forwards=0)
        original_write(path, data)
    engine.inspect, engine.accept, engine.select, engine._write = guard.inspect_guard, guard.accept_guard, make_selector(), write_variant
    runtime._predict_logits = predict_logged
    try:
        return engine.run(OUTPUT)
    finally:
        engine.inspect, engine.accept, engine.select, engine._write = original_inspect, original_accept, original_select, original_write
        runtime._predict_logits = original_predict


def self_test() -> int:
    """Top-1/5의 약한 대체 조건·후보 전환 반례·고정 floor·64개 quota를 모델 학습 없이 검사한다."""
    y = np.zeros(2048, dtype=np.int64)
    z = np.full((2048, 372), -100., dtype=np.float32); z[:, 1] = 0.; z[:, 0] = 2.
    z[:4, 0] = [.1, .2, .3, .4]; z[40, 0] = 1.01; z[40, 2] = -.001
    z[1040, :7] = [.5, 2., 1.5, 1., .8, .4, .399]
    spec = {"top1_mask": np.ones(2048, dtype=bool), "top1_floor": np.ones(2048, dtype=np.float32),
            "top5_mask": np.zeros(2048, dtype=bool), "top5_floor": np.zeros(2048, dtype=np.float32)}
    spec["top1_mask"][1040] = False; spec["top5_mask"][1040] = True; spec["top5_floor"][1040] = .09
    items = make_selector()(z, y, spec, guard.inspect_guard(z, y, spec), [dict(row=40, rank=1, deficit=.001), dict(row=1040, rank=5, deficit=.001)])
    assert any(i["row"] == 40 and i["rank"] == 1 and i["rival"] == 2 and i["role"] == "priority_near_alternative_rival" for i in items)
    assert any(i["row"] == 1040 and i["rank"] == 5 and i["rival"] == 6 and i["role"] == "priority_near_alternative_rival" for i in items)
    switched = z.copy(); switched[40, 2] = 1.2
    assert switched[40, 0]-switched[40, 1] >= spec["top1_floor"][40]
    assert switched[40, 0]-switched[40, 2] < spec["top1_floor"][40]
    assert len(items) <= 64 and len({(i["row"], i["rank"], i["rival"]) for i in items}) == len(items)
    assert sum(i["role"] == "priority_near_alternative_rival" for i in items) <= ALTERNATIVE_CAP
    assert all(i["floor"] == float(spec[f'top{i["rank"]}_floor'][i["row"]]) for i in items)
    print(json.dumps(dict(self_test="pass", weak_top1_and_top5_alternatives_selected=True,
        current_rival_only_can_miss_switched_rival=True, original_floors_unchanged=True, selected=len(items))))
    return 0


def main() -> int:
    """기존 봉인 파일을 변경하지 않고 새 비교 실험 또는 작은 수치 반례 검사에 진입한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
