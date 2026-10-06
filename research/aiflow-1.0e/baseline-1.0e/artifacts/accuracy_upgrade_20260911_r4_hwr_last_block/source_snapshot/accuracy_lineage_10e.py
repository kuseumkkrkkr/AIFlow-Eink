"""실제 fit/selection ID와 파일 SHA를 추적하는 연구용 감독학습 계보."""
from __future__ import annotations

import json
from pathlib import Path

from accuracy_upgrade_contract_v1 import canonical_json_sha256, normalize_latex, sha256_file


class LineageRegistry:
    """모든 데이터 ID와 학습 산출물의 부모 관계를 한 manifest에 기록한다."""

    def __init__(self, records: dict[str, dict], nodes: dict[str, dict] | None = None):
        """writer·수식·raw SHA가 고정된 데이터 색인을 받는다."""
        self.records = dict(records)
        self.nodes = nodes or {}
        self._verified: set[tuple[str, str]] = set()

    def add(self, name: str, artifact: Path, fit_ids=(), parents=(), selection_ids=(), **metadata) -> str:
        """학습기가 실제 사용한 ID와 저장된 파일의 SHA를 등록한다."""
        fit_ids, selection_ids = sorted(set(fit_ids)), sorted(set(selection_ids))
        unknown = (set(fit_ids) | set(selection_ids)) - self.records.keys()
        if unknown or set(parents) - self.nodes.keys():
            raise ValueError(f"unregistered training IDs or parents: {sorted(unknown)[:3]}")
        node = dict(name=name, artifact_path=str(artifact.resolve()), artifact_sha256=sha256_file(str(artifact)),
                    fit_record_ids=fit_ids, selection_record_ids=selection_ids, parents=sorted(parents), **metadata)
        node_id = canonical_json_sha256(node)
        self.nodes[node_id] = node
        return node_id

    def validate(self, root: str, excluded_writers: set[str], excluded_formulas: set[str], verify_files: bool = True) -> set[str]:
        """조상 전체의 학습·모델선택 자료와 실물 파일을 확인한다."""
        visited, active, used = set(), set(), set()

        def visit(node_id: str) -> None:
            """부모 노드를 재귀 검사하며 순환과 자기 선언식 provenance를 거부한다."""
            if node_id in active:
                raise ValueError("lineage cycle")
            if node_id in visited:
                return
            node = self.nodes.get(node_id)
            if not node or canonical_json_sha256(node) != node_id:
                raise ValueError("missing or modified lineage node")
            active.add(node_id)
            if verify_files:
                key = (node["artifact_path"], node["artifact_sha256"])
                if key not in self._verified:
                    if sha256_file(key[0]) != key[1]:
                        raise ValueError("lineage checkpoint SHA mismatch")
                    self._verified.add(key)
            for record_id in node["fit_record_ids"] + node["selection_record_ids"]:
                record = self.records.get(record_id)
                if not record:
                    raise ValueError("unknown training record")
                if record["writer"] in excluded_writers:
                    raise ValueError(f"writer leakage: {record['writer']}")
                if record["formula_key"] in excluded_formulas:
                    raise ValueError("formula leakage")
                used.add(record_id)
            for parent in node["parents"]:
                visit(parent)
            active.remove(node_id)
            visited.add(node_id)

        visit(root)
        return used

    def payload(self) -> dict:
        """manifest 직렬화와 cache key용 고정 JSON을 반환한다."""
        return dict(schema="aiflow-training-lineage/v2", records=self.records, nodes=self.nodes,
                    records_sha256=canonical_json_sha256(self.records))

    @classmethod
    def load(cls, path: Path, expected_sha: str | None = None):
        """manifest SHA와 데이터 색인의 무결성을 확인한다."""
        if expected_sha and sha256_file(str(path)) != expected_sha:
            raise ValueError("lineage manifest SHA mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "aiflow-training-lineage/v2":
            raise ValueError("strict evaluation requires lineage/v2")
        if canonical_json_sha256(payload["records"]) != payload["records_sha256"]:
            raise ValueError("lineage record index mismatch")
        return cls(payload["records"], payload["nodes"])


def validate_prediction(row: dict, outer_writer: str, registry: LineageRegistry | None = None) -> None:
    """예측 자신과 outer writer 및 평가 수식이 모든 감독학습 조상에서 빠졌는지 검사한다."""
    provenance = row.get("prediction_provenance") or {}
    if registry is None:
        if not provenance.get("registry_path") or not provenance.get("registry_sha256"):
            raise ValueError("strict prediction requires actual lineage registry")
        registry = LineageRegistry.load(Path(provenance["registry_path"]), provenance["registry_sha256"])
    record = registry.records.get(str(row["record_id"]))
    writer = str(row.get("raw_writer_group") or row["writer_group"])
    if not record or record["writer"] != writer:
        raise ValueError("prediction writer/record lineage mismatch")
    expected = provenance.get("candidate_sha256")
    if expected != canonical_json_sha256(row["candidates"]):
        raise ValueError("prediction candidate hash mismatch")
    excluded_formulas = {record["formula_key"]}
    excluded_formulas.update(r["formula_key"] for r in registry.records.values() if r["writer"] == outer_writer)
    registry.validate(provenance.get("root_id", ""), {writer, str(outer_writer)}, excluded_formulas)


def filtered_ids(records: dict[str, dict], allowed: set[str], held: set[str]) -> list[str]:
    """held writer와 동일 수식의 학습 records를 동시에 제거한다."""
    writers = {records[key]["writer"] for key in held}
    formulas = {records[key]["formula_key"] for key in held}
    return sorted(key for key in allowed if records[key]["writer"] not in writers
                  and records[key]["formula_key"] not in formulas)
