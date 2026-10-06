from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_public_dataset09 import build


class BuildPublicDatasetReplayTest(unittest.TestCase):
    def test_matches_reidentified_replay_by_original_hash_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "archive" / "samples" / "bucket" / "sample.json"
            sample.parent.mkdir(parents=True)
            old_session = "legacy-session"
            old_prompt = "legacy-prompt"
            sample.write_text(json.dumps({
                "session_id": "reidentified-session",
                "contributor_id": "reidentified-writer",
                "prompt_id": "reidentified-prompt",
                "target_display": "x",
                "target_cells": [{"token": "x"}],
                "target_relations": [],
                "formula_cells": [],
                "ownership_status": "unreviewed",
                "label_status": "confirmed",
                "canvas": {"width": 100, "height": 100},
                "strokes": [{"order": 0, "points": [{"x": 1, "y": 2, "t_ms": 3}]}],
                "source": "owned-phone-replay",
                "original_session_hash": sha256(old_session.encode("utf-8")).hexdigest(),
                "original_prompt_id": old_prompt,
            }), encoding="utf-8")

            arrival = root / "arrival.json"
            arrival.write_text(json.dumps({"records": {
                "samples/bucket/sample.json": {"reviewStatus": "reviewed", "decision": "valid"}
            }}), encoding="utf-8")
            ownership = root / "ownership.json"
            ownership.write_text(json.dumps({"rows": []}), encoding="utf-8")
            replay_source = root / "replay.jsonl"
            replay_source.write_text(json.dumps({
                "session_id": old_session,
                "prompt_id": old_prompt,
                "expected": ["x"],
            }) + "\n", encoding="utf-8")
            replay_annotations = root / "replay_annotations.json"
            replay_annotations.write_text(json.dumps({
                "groups": {old_prompt: [[0]]},
                "excluded": {},
            }), encoding="utf-8")

            output = root / "output"
            manifest = build(Namespace(
                archive_root=root / "archive",
                arrival_index=arrival,
                ownership_annotations=ownership,
                replay_source=replay_source,
                replay_annotations=replay_annotations,
                output=output,
            ))

            self.assertEqual(manifest["ownership_replay_formulas"], 1)
            self.assertEqual(manifest["ownership_replay_match"], {
                "direct": 0,
                "original_session_sha256_prompt": 1,
                "unmatched": 0,
            })
            replay = json.loads((output / "data" / "ownership_replay.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(replay["labels"], ["x"])
            self.assertEqual(replay["groups"], [[0]])


if __name__ == "__main__":
    unittest.main()
