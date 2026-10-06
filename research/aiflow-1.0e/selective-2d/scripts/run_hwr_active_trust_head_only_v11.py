"""봉인 v7의 선택·예산을 유지하고 encoder gradient만 끊어 선형 head 교정을 비교한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hwr_active_trust_rival_guard_v7 as previous

engine = previous.engine
OUTPUT = engine.ROOT / "artifacts/hwr_active_trust_retention_20261005_head_only_v11"
HEAD_NAMES = ("math_head.weight", "math_head.bias")


def detach_encoder(model):
    """encode 값은 바꾸지 않고 반환 embedding의 gradient 연결만 끊어 head 밖 미분을 차단한다."""
    original_encode = model.encode
    def encode_detached(x):
        """실제 encoder를 그대로 호출하며 head 입력 값·모양·dtype는 유지한다."""
        return original_encode(x).detach()
    model.encode = encode_detached


def run() -> int:
    """원 parent에서 시작해 모든 gradient·실제 forward의 head 및 encoder 불변 증거를 저장한다."""
    import torch
    import run_hwr_affine_distillation_experiment_v1 as runtime
    plan = json.loads((previous.OUTPUT / "frozen_plan.json").read_text(encoding="utf-8"))
    verified = json.loads((previous.OUTPUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert engine._sha(Path(previous.__file__)) == plan["entrypoint_sha256"]
    assert engine._sha(previous.OUTPUT / "active_trust_result.json") == verified["result_sha256"]
    original_output, original_write = previous.OUTPUT, engine._write
    original_load, original_predict, original_grad = runtime._load_teacher, runtime._predict_logits, torch.autograd.grad
    state = {}; gradient_records = []; head_trace = []; pending = []
    def load_detached(path, device):
        """parent에만 값 변경 없는 detach를 적용하며 55개 비-head tensor의 최초 비트를 보관한다."""
        model, labels, report = original_load(path, device)
        if Path(path).resolve() == (engine.CANDIDATE / "directional_guard.pt").resolve():
            assert isinstance(model.math_head, torch.nn.Linear)
            assert (model.math_head.in_features, model.math_head.out_features) == (128, 372)
            named = dict(model.named_parameters()); assert len(named) == 57
            assert tuple(n for n in named if n.startswith("math_head.")) == HEAD_NAMES
            assert all(p.dtype == torch.float32 for p in named.values())
            state.update(model=model, names={id(p): n for n,p in named.items()},
                frozen={n:p.detach().clone() for n,p in named.items() if n not in HEAD_NAMES})
            assert len(state["frozen"]) == 55
            detach_encoder(model)
        return model, labels, report
    def grad_checked(outputs, inputs, *args, **kwargs):
        """기존 미분을 한 번만 호출하고 55개 unused 및 두 head의 실제 norm을 검사한다."""
        result = original_grad(outputs, inputs, *args, **kwargs)
        names = [state["names"][id(p)] for p in inputs]
        assert len(names) == 57
        assert all(g is None for n,g in zip(names,result) if n not in HEAD_NAMES)
        norms = {n:float(g.detach().double().norm()) for n,g in zip(names,result) if n in HEAD_NAMES}
        assert tuple(norms) == HEAD_NAMES and all(np.isfinite(v) and v > 0 for v in norms.values())
        gradient_records.append(dict(index=len(gradient_records), unused_nonhead_tensors=55, head_norms_fp64=norms))
        return result
    def save_head(item, weight, bias):
        """기존 forward와 동시 관측한 작은 head tensor만 NPY와 hash로 봉인한다."""
        for name, array in (("weight",weight),("bias",bias)):
            np.save(OUTPUT / item[f"{name}_file"], array, allow_pickle=False)
            item[f"{name}_sha256"] = engine._sha(OUTPUT / item[f"{name}_file"])
    def predict_checked(model, x, device, batch):
        """원래 full forward 뒤 55개 tensor의 int32 비트 불변과 당시 head 상태를 추가 호출 없이 기록한다."""
        actual = original_predict(model, x, device, batch)
        assert model is state["model"]
        named = dict(model.named_parameters())
        assert all(torch.equal(named[n].detach().view(torch.int32), p.view(torch.int32)) for n,p in state["frozen"].items())
        index = len(head_trace); assert index < 2+engine.ROUNDS*len(engine.ALPHAS)
        item = dict(index=index, weight_file=f"head_weight_{index:03d}.npy", bias_file=f"head_bias_{index:03d}.npy",
            frozen_nonhead_tensors=55, all_nonhead_int32_bit_equal=True)
        head_trace.append(item)
        weight = named[HEAD_NAMES[0]].detach().cpu().numpy().copy()
        bias = named[HEAD_NAMES[1]].detach().cpu().numpy().copy()
        if not OUTPUT.exists():
            assert index == 0; pending.append((item,weight,bias))
        else:
            for saved in pending:
                save_head(*saved)
            pending.clear(); save_head(item,weight,bias)
        return actual
    def write_variant(path, data):
        """gradient 영역만 바꾼 v11 계약과 회차별 head·encoder 증거를 새 실험 출력에 봉인한다."""
        if path == OUTPUT / "frozen_plan.json":
            data.update(entrypoint_file=Path(__file__).name, entrypoint_sha256=engine._sha(Path(__file__)),
                head_only_revision="v11", head_parameter_names=list(HEAD_NAMES), frozen_nonhead_tensors=55,
                head_contract=dict(type="Linear", in_features=128, out_features=372, bias=True),
                predecessor_result_sha256=verified["result_sha256"],
                changed_factors="only encoder gradient detachment versus v7; actual encoder values, v7 selection, original floors, guards, solver, alphas, 64/512 direction caps and forwards unchanged",
                optimization_scope="two linear-head tensors only; all 55 encoder tensors bitwise checked at every real score forward",
                evidence_capture_extra_model_or_gradient_calls=0)
            data["dependencies"][Path(previous.__file__).name] = engine._sha(Path(previous.__file__))
        if path == OUTPUT / "active_trust_result.json":
            assert not pending and len(gradient_records) == data["gradient_directions_used"]
            assert len(head_trace) == data["actual_model_checks"]+2
            data["head_only_trace"] = dict(gradient_records=gradient_records, head_parameter_trace=head_trace,
                frozen_nonhead_tensors=55, evidence_capture_extra_model_or_gradient_calls=0)
        original_write(path, data)
    previous.OUTPUT, engine._write = OUTPUT, write_variant
    runtime._load_teacher, runtime._predict_logits, torch.autograd.grad = load_detached, predict_checked, grad_checked
    try:
        return previous.run()
    finally:
        previous.OUTPUT, engine._write = original_output, original_write
        runtime._load_teacher, runtime._predict_logits, torch.autograd.grad = original_load, original_predict, original_grad


def self_test() -> int:
    """작은 nonlinear encoder에서 forward 값 불변·unused gradient·선형 head 실제 응답을 검사한다."""
    import torch
    torch.manual_seed(11); torch.set_num_threads(2)
    class Toy(torch.nn.Module):
        """실제 실험 데이터 없이 detach의 숫자 보존과 미분 범위를 검증하는 작은 모델이다."""
        def __init__(self):
            """5채널 입력을 128차원 nonlinear embedding 및 372-class head로 구성한다."""
            super().__init__(); self.encoder=torch.nn.Linear(5,128); self.math_head=torch.nn.Linear(128,372)
        def encode(self,x):
            """tanh nonlinear encoder를 반환하여 head만 미분하는 조건을 검사한다."""
            return torch.tanh(self.encoder(x))
    model=Toy(); x=torch.randn(1,5)
    before=model.math_head(model.encode(x)).detach(); detach_encoder(model)
    embedding=model.encode(x); actual=model.math_head(embedding)
    assert torch.equal(before,actual.detach()) and not embedding.requires_grad
    named=list(model.named_parameters()); margin=actual[0,2]-actual[0,3]
    grads=torch.autograd.grad(margin,[p for _,p in named],allow_unused=True)
    assert all(g is None for (n,_),g in zip(named,grads) if not n.startswith("math_head."))
    assert all(g is not None for (n,_),g in zip(named,grads) if n.startswith("math_head."))
    predicted=float(margin.detach())+1e-3*sum(float(g.double().square().sum()) for g in grads if g is not None)
    with torch.no_grad():
        for (_,p),g in zip(named,grads):
            if g is not None: p.add_(1e-3*g)
    values=model.math_head(model.encode(x)); residual=abs(float(values[0,2]-values[0,3])-predicted)
    assert residual < 1e-5
    print(json.dumps(dict(self_test="pass", encoder_values_bit_equal=True, encoder_gradients_unused=True,
        head_only_actual_margin_residual=residual)))
    return 0


def main() -> int:
    """별도 봉인 실험 또는 데이터 없는 gradient 경계 검증에 진입한다."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("run","self-test"),required=True)
    return self_test() if parser.parse_args().mode == "self-test" else run()


if __name__ == "__main__":
    raise SystemExit(main())
