"""Explicit historical audit only. Never imported by the active research runtime."""
import hashlib
import json
from pathlib import Path

import numpy as np
from reconstructed_mini_lm_numpy_v63 import NumpyLM, load_checkpoint

HERE=Path(__file__).resolve().parent
ARTIFACT=HERE.parent
ROOT=HERE.parents[2]
manifest=json.loads((HERE/'ARCHIVE_MANIFEST.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
checkpoint=ROOT/'artifacts/hwr_failure_cause_microscope_20260928/mini_lm_distill_20261003_one_layer/mini_formula_lm.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='2c448199c2b4d65c4d786348ad2f3ba09322fedb5337b35d503f48d0462d01f2'
model=NumpyLM(load_checkpoint(checkpoint))
requests=[json.loads(s) for s in (ARTIFACT/'requests.jsonl').read_text().splitlines()]
logits=np.stack([model.forward(r['ids'],r['target']) for r in requests])
output=ARTIFACT/'migration_reference_logits.npz'
if output.exists():
    saved=np.load(output,allow_pickle=False)
    assert np.array_equal(saved['logits'],logits)
    assert saved['record_ids'].tolist()==[r['record_id'] for r in requests]
    print('Existing same-input reference verified')
else:
    np.savez(output,logits=logits,record_ids=np.asarray([r['record_id'] for r in requests]))
    print('Historical same-input reference generated')
