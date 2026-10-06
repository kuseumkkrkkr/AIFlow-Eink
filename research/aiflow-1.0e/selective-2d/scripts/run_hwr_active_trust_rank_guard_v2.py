"""봉인 active-trust 엔진에서 실제 정답 순위 보존을 margin 보존과 별도로 강화한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_retention_v1 as engine

OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_rank_guard_v2"
BASE_INSPECT, BASE_ACCEPT = engine.inspect, engine.accept


def inspect_guard(logits, y, spec):
    """floor 위반 여부와 별개로 보호 대상의 현재 Top-1/Top-5 정답 마스크를 저장한다."""
    result = BASE_INSPECT(logits, y, spec)
    result["rank_guard"] = {k: spec[f"top{k}_mask"].astype(bool) & result["rank"][f"top{k}_mask"] for k in (1, 5)}
    return result


def accept_guard(before, after):
    """정답 순위 손실과 정상 floor 손실을 합쳐 거부하며 원래 실제 merit 감소도 유지한다."""
    allowed, reason, floor_losses = BASE_ACCEPT(before, after)
    lost = {(item["row"], item["rank"]): item for item in floor_losses}
    rank_losses = [item for item in after["failures"] if before["rank_guard"][item["rank"]][item["row"]]
                   and not after["rank"][f'top{item["rank"]}_mask'][item["row"]]]
    if rank_losses:
        lost.update({(item["row"], item["rank"]): item for item in rank_losses})
        return False, "previously_correct_protected_rank_loss", list(lost.values())
    return allowed, reason, floor_losses


def run() -> int:
    """원 v1 입력·예산을 그대로 쓰며 실제 rank guard 변경을 plan 저장 전에 봉인한다."""
    original = json.loads((engine.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(engine.__file__)) == original["script_sha256"]
    original_write = engine._write
    def write_variant(path, data):
        """별도 v2 plan에 진입점·정답 순위 보존·원본 실패 증거 hash를 함께 기록한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)), rank_guard_revision="v2",
                predecessor_result_sha256=engine._sha(engine.OUTPUT / "active_trust_result.json"),
                actual_step_acceptance="no currently correct protected Top-1/5 rank loss AND no currently safe floor loss AND actual merit decrease AND norm cap",
                changed_factors="v1 active/trust method plus separate current-rank guard; no original floor change")
        original_write(path, data)
    engine.inspect, engine.accept, engine._write = inspect_guard, accept_guard, write_variant
    try:
        return engine.run(OUTPUT)
    finally:
        engine.inspect, engine.accept, engine._write = BASE_INSPECT, BASE_ACCEPT, original_write


def self_test() -> int:
    """floor가 이미 나쁜 정답 행의 순위 손실을 오차 감소로 은폐할 수 없음을 검사한다."""
    y = np.zeros(3, dtype=np.int64); before_logits = np.full((3, 372), -100., dtype=np.float32)
    before_logits[:, 1] = 0.; before_logits[:, 0] = [.5, -2., 2.]
    spec = {f"top{k}_{name}": value for k in (1, 5) for name, value in (
        ("mask", np.array([True, True, False])),
        ("floor", np.ones(3, dtype=np.float32) if k == 1 else np.zeros(3, dtype=np.float32)))}
    before = inspect_guard(before_logits, y, spec)
    unsafe_logits = before_logits.copy(); unsafe_logits[:, 0] = [-.1, .9, 2.]
    after = inspect_guard(unsafe_logits, y, spec)
    assert BASE_ACCEPT(before, after)[0] and after["merit"] < before["merit"]
    allowed, reason, losses = accept_guard(before, after)
    assert not allowed and reason == "previously_correct_protected_rank_loss" and len(losses) == 1
    safe_logits = before_logits.copy(); safe_logits[1, 0] = -.5
    assert accept_guard(before, inspect_guard(safe_logits, y, spec))[0]
    print(json.dumps(dict(self_test="pass", rank_loss_hidden_by_deficit_improvement_rejected=True,
        safe_rank_preserving_progress_allowed=True, original_floors_unchanged=True)))
    return 0


def main() -> int:
    """기존 증거를 덮지 않고 별도 rank-guard 실행 또는 해당 반례 검증만 수행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "self-test"), required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
