import copy
import unittest

from validate_accuracy_collection_10e import validate_collection


class CollectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "writer_sessions": 2,
            "roles": {
                "train": {"writers": 1, "formulas": 2},
                "sealed_acceptance": {"writers": 1, "formulas": 2},
            },
            "formula_mix_percent": {"arithmetic": 50, "scripts": 50},
            "required_fields": [
                "record_id", "writer_id", "session_id", "device_id", "prompt_id",
                "strokes", "stroke_times", "symbol_ownership", "symbol_labels",
                "structure_relations", "latex", "annotation_status", "rights_status",
            ],
            "rules": {
                "writer_disjoint": True,
                "prompt_disjoint": True,
                "device_disjoint_in_sealed_acceptance": True,
            },
        }
        self.records = []
        for role, writer, device, prompt_prefix in [
            ("train", "w-train", "d-train", "p-train"),
            ("sealed_acceptance", "w-sealed", "d-sealed", "p-sealed"),
        ]:
            for index, category in enumerate(["arithmetic", "scripts"]):
                self.records.append({
                    "record_id": f"{role}-{index}", "role": role, "category": category,
                    "writer_id": writer, "session_id": f"s{index + 1}", "device_id": device,
                    "prompt_id": f"{prompt_prefix}-{index}", "strokes": [[[0, 0]]],
                    "stroke_times": [[0]], "symbol_ownership": [{"stroke_indices": [0]}],
                    "symbol_labels": ["1"], "structure_relations": [], "latex": "1",
                    "annotation_status": "independently_reviewed", "rights_status": "approved",
                })

    def test_valid_collection_passes(self):
        self.assertEqual(validate_collection(self.contract, self.records)["status"], "passed")

    def test_overlap_and_duplicate_ownership_fail(self):
        rows = copy.deepcopy(self.records)
        rows[2]["writer_id"] = "w-train"
        rows[0]["symbol_ownership"] = [
            {"stroke_indices": [0]}, {"stroke_indices": [0]},
        ]
        result = validate_collection(self.contract, rows)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("writer overlap" in error for error in result["errors"]))
        self.assertTrue(any("exactly once" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
