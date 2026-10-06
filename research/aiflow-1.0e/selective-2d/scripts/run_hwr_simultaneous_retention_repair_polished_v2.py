"""봉인 v1 엔진을 보존하며 dual 활성 집합의 FP64 정밀 보정만 추가한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_simultaneous_retention_repair_v1 as engine

OUTPUT = engine.ROOT / "artifacts/hwr_simultaneous_retention_repair_20261005_polished_v2"
BASE_SOLVER = engine.solve_gram


def solve_polished(gram: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, dict]:
    """기존 solver의 양수 dual 집합을 1회 정확히 풀고 동일 잔차 기준으로 재인증한다."""
    coefficients, before = BASE_SOLVER(gram, rhs)
    active = np.flatnonzero(coefficients > 1e-10)
    polished = coefficients.copy()
    solved = False
    if len(active):
        try:
            polished[:] = 0.
            polished[active] = np.linalg.solve(gram[np.ix_(active, active)], rhs[active])
            solved = True
        except np.linalg.LinAlgError:
            polished = coefficients
    residual = gram @ polished - rhs
    stationarity = np.where(polished > 1e-10, np.abs(residual), np.maximum(-residual, 0.))
    certificate = dict(before_polish=before, active_constraints=len(active), active_linear_solve_completed=solved,
        solver_success=before["solver_success"], solver_iterations=before["solver_iterations"],
        solver_message="v1 dual plus one FP64 active-set linear solve; no tolerance change",
        min_primal_residual=float(residual.min()), max_stationarity_residual=float(stationarity.max()),
        certified=bool(np.isfinite(polished).all() and (polished >= 0).all()
                       and residual.min() >= -1e-9 and stationarity.max() <= 1e-8))
    return polished, certificate


def run(out: Path) -> int:
    """동일 엔진·예산·제약을 사용하고 시작 전 manifest에 실제 v2 의존성을 봉인한다."""
    original = json.loads((engine.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(engine.__file__)) == original["script_sha256"]
    original_write = engine._write
    def write_variant(path, data):
        """plan 저장 전에 solver 교체와 실행 진입점을 명시하며 결과 파일은 그대로 기록한다."""
        if path == out / "frozen_plan.json":
            data.update(entrypoint_sha256=engine._sha(Path(__file__)),
                solver="v1 dual <=2000 iterations plus one FP64 active-set linear solve; unchanged residual certificate",
                solver_revision="polished_v2", previous_numerical_failure_result_sha256=engine._sha(engine.OUTPUT / "simultaneous_result.json"))
        original_write(path, data)
    # 이미 봉인한 v1 소스·실험은 수정하지 않는다. 이 단일 프로세스에서만 명시적 교체한다.
    engine.solve_gram, engine._write = solve_polished, write_variant
    try:
        return engine.run(out)
    finally:
        engine.solve_gram, engine._write = BASE_SOLVER, original_write


def self_test() -> int:
    """실제 실패 Gram의 정밀 보정과 가능한/불가능한 toy의 인증을 함께 검사한다."""
    sample = np.load(engine.OUTPUT / "linearized_round_000.npz", allow_pickle=False)
    _, certificate = solve_polished(sample["gram"], sample["rhs"])
    assert certificate["certified"] and certificate["before_polish"]["min_primal_residual"] < -1e-9
    original_solver = engine.solve_gram
    engine.solve_gram = solve_polished
    try:
        engine.self_test()
    finally:
        engine.solve_gram = original_solver
    print(json.dumps(dict(recorded_failed_gram_now_certified=True,
        min_primal_residual=certificate["min_primal_residual"], tolerance_unchanged=True)))
    return 0


def main() -> int:
    """별도 v2 출력에서 실행하며 v1 실패 증거의 덮어쓰기를 허용하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "run"), required=True)
    args = parser.parse_args()
    return self_test() if args.mode == "self-test" else run(OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
