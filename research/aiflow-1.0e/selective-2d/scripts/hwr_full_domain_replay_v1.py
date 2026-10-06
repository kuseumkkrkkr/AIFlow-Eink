"""전 범위 TRAIN 변형의 봉인 검증·클래스/위상 균등 replay를 제공한다.

teacher 분포는 연구용 의사 표적일 뿐 사람 경계 라벨이 아니다. 이 모듈은
학습이나 승격을 실행하지 않으며 기존 class-2 실험의 봉인 코드를 수정하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_hwr_full_domain_tube_v1 import _geometry, _signature
from run_hwr_pendigits_tube_probe_v1 import _sha


def load_full_domain_candidates(report_path: Path, checkpoint_sha: str, labels: list[str]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """공식 TRAIN·봉인 SHA·행별 원본/변형 대응이 맞는 연구 후보만 읽는다."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiflow-hwr-full-domain-tube-audit/v1":
        raise ValueError("unknown full-domain report schema")
    p = report["provenance"]
    if p["teacher_checkpoint_sha256"] != checkpoint_sha or p["heldout_rows_read"] or p["crohme_rows"] or p["official_test_rows_read"] or p["real_training_source_ids"] != [0, 1]:
        raise ValueError("candidate teacher/source policy differs")
    if _sha(report_path.parent / "frozen_plan.json") != report["frozen_plan_sha256"]:
        raise ValueError("candidate plan changed")
    cache = report["candidate_cache"]
    if len(labels) != 372 or len(set(labels)) != 372 or cache["class_labels"] != labels:
        raise ValueError("candidate class order differs")
    arrays = {}
    for name in ("candidate_features", "teacher_logits"):
        item = cache["artifacts"][name]; path = report_path.parent / f"{name}.npy"
        if str(path.resolve()) != item["path"] or _sha(path) != item["sha256"]:
            raise ValueError("candidate cache path/SHA differs")
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]:
            raise ValueError("candidate cache dtype/shape differs")
        arrays[name] = value
    mapping_path = report_path.parent / "blind_review_mapping.json"
    if _sha(mapping_path) != report["review_packet_sha256"][mapping_path.name]:
        raise ValueError("candidate provenance mapping changed")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    x, z = arrays["candidate_features"], arrays["teacher_logits"]
    n = len(mapping)
    if not n or n % 2 or x.shape != (n, 128, 5) or z.shape != (n, 372) or x.dtype != np.float32 or z.dtype != np.float32:
        raise ValueError("candidate arrays are not aligned float32 pairs")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("nonfinite candidate input")
    classes = []
    for i in range(0, n, 2):
        first, second = mapping[i:i + 2]
        if first["candidate_row"] != i or second["candidate_row"] != i + 1 or first["endpoint"] != "original" or second["endpoint"] != "deformed":
            raise ValueError("candidate pair order differs")
        if any(first[key] != second[key] for key in ("class_id", "query_training_row", "donor_training_row")) or not 0 <= first["class_id"] < 372:
            raise ValueError("candidate pair provenance differs")
        if _signature(x[i]) != _signature(x[i + 1]) or not _geometry(x[i], x[i + 1])["valid"]:
            raise ValueError("candidate geometry integrity failed")
        classes.append(first["class_id"])
    return report, x.reshape(-1, 2, 128, 5), z.reshape(-1, 2, 372), np.asarray(classes, dtype=np.int64)


class ClassTopologyBalancedSampler:
    """클래스를 중복 없이 순회하고 각 클래스 안에서도 획 위상을 순회한다."""
    def __init__(self, class_ids: np.ndarray, originals: np.ndarray, seed: int):
        """별도 RNG/위상별 후보 인덱스를 만들어 기존 학습 난수열을 건드리지 않는다."""
        if class_ids.shape != (len(originals),) or not len(originals):
            raise ValueError("sampler needs aligned nonempty originals/class IDs")
        self.rng = np.random.default_rng(seed)
        self.groups = {}
        for i, class_id in enumerate(class_ids):
            if not 0 <= class_id < 372:
                raise ValueError("class ID out of range")
            self.groups.setdefault(int(class_id), {}).setdefault(_signature(originals[i]), []).append(i)
        self.class_order, self.topology_order = [], {c: [] for c in self.groups}
        self.pair_order = {(c, t): [] for c, group in self.groups.items() for t in group}

    def sample(self, count: int) -> np.ndarray:
        """각 클래스/위상의 동일 예산을 유지하면서 count개 pair ID를 반환한다."""
        if count < 1:
            raise ValueError("sample count must be positive")
        result = []
        for _ in range(count):
            if not self.class_order:
                self.class_order = self.rng.permutation(sorted(self.groups)).tolist()
            c = self.class_order.pop()
            if not self.topology_order[c]:
                keys = list(self.groups[c])
                self.topology_order[c] = [keys[i] for i in self.rng.permutation(len(keys))]
            t = self.topology_order[c].pop()
            if not self.pair_order[(c, t)]:
                self.pair_order[(c, t)] = self.rng.permutation(self.groups[c][t]).tolist()
            result.append(self.pair_order[(c, t)].pop())
        return np.asarray(result, dtype=np.int64)


def class_macro_teacher_kl(student_logits, teacher_logits, class_ids, temperature: float = 2.):
    """각 클래스 평균을 다시 평균하여 표본이 많은 클래스의 과도한 기여를 막는다."""
    import torch
    from torch.nn import functional as F
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2 or student_logits.shape[1] != 372 or not len(student_logits):
        raise ValueError("full-domain loss requires aligned nonempty [N,372] logits")
    if class_ids.shape != (len(student_logits),) or class_ids.dtype != torch.long or not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("full-domain class IDs/logits invalid")
    if (class_ids < 0).any() or (class_ids >= 372).any() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("class ID/temperature out of range")
    target = torch.softmax(teacher_logits.detach().float() / temperature, 1)
    rows = F.kl_div(F.log_softmax(student_logits.float() / temperature, 1), target, reduction="none").sum(1) * temperature ** 2
    return torch.stack([rows[class_ids == c].mean() for c in torch.unique(class_ids)]).mean()


def _self_test() -> None:
    """클래스/위상 비중·별도 RNG·macro gradient와 teacher detach를 확인한다."""
    import torch
    u = np.linspace(.1, .9, 128, dtype=np.float32)
    x = np.zeros((12, 128, 5), dtype=np.float32)
    x[:, :, 0] = u; x[:, :, 1] = .5; x[:, 0, 3] = 1; x[:, :, 4] = 1
    x[8:10, 64, 3] = 1
    classes = np.array([0] * 10 + [1, 2])
    a = ClassTopologyBalancedSampler(classes, x, 17); b = ClassTopologyBalancedSampler(classes, x, 17)
    chosen = a.sample(300)
    assert np.array_equal(chosen, b.sample(300))
    assert np.array_equal(np.bincount(classes[chosen], minlength=3), [100, 100, 100])
    class0 = chosen[classes[chosen] == 0]
    assert sum(len(_signature(x[i])) == 2 for i in class0) == 50
    student = torch.zeros(11, 372, requires_grad=True)
    teacher = torch.ones(11, 372, requires_grad=True)
    with torch.no_grad():
        teacher[:, 0] = 4
    loss = class_macro_teacher_kl(student, teacher, torch.tensor([0] * 10 + [1]))
    loss.backward()
    assert teacher.grad is None
    assert torch.allclose(student.grad[:10].sum(0), student.grad[10], atol=1e-6, rtol=1e-5)
    print(json.dumps(dict(self_test="pass", class_budget_exact=True, topology_budget_exact=True, deterministic=True, macro_gradient_equal=True)))


if __name__ == "__main__":
    _self_test()
