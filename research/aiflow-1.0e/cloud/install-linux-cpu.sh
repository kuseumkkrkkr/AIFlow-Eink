#!/usr/bin/env bash
set -euo pipefail
cd /workspace/AIFlow-Eink
test -f README.md
test -f research/contracts/aiflow_p_formula_v1.schema.json
setup_dir=/workspace/.aiflow-eink-env
mkdir -p "$setup_dir"
python3 -m venv "$setup_dir/venv"
PIP_CACHE_DIR="$setup_dir/pip-cache" "$setup_dir/venv/bin/python" -m pip install --disable-pip-version-check \
  attrs==26.1.0 jsonschema==4.25.1 jsonschema-specifications==2025.9.1 \
  referencing==0.37.0 rpds-py==2026.9.1 typing_extensions==4.16.0
cat > "$setup_dir/validate.py" <<'AIFLOW_VALIDATION_PY'
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

ROOT = Path('/workspace/AIFlow-Eink')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


documents = {}
for path in sorted(p for folder in (ROOT / 'models', ROOT / 'research/contracts', ROOT / 'research/configs', ROOT / 'research/reports') for p in folder.rglob('*.json')):
    documents[path.relative_to(ROOT).as_posix()] = json.loads(path.read_text(encoding='utf-8'))
require(bool(documents), 'No JSON documents found')
print(f'PASS: parsed {len(documents)} JSON documents')

schemas = {
    path: document for path, document in documents.items()
    if path.startswith('research/contracts/') and path.endswith('.schema.json')
}
require(len(schemas) == 2, 'Expected both published formula schemas')
for schema in schemas.values():
    Draft202012Validator.check_schema(schema)
print(f'PASS: {len(schemas)} Draft 2020-12 schema definitions')

record = {
    'formula_id': 'synthetic-setup', 'origin_id': 'synthetic-origin',
    'writer_id': 'synthetic-writer', 'device_id': 'synthetic-device',
    'source_id': 'synthetic-source', 'split': 'training',
    'canvas_width': 128, 'canvas_height': 128, 'rights_track': 'P',
    'commercial_training_allowed': True,
    'symbols': [{'token': 'x', 'strokes': [[{'x': 10, 'y': 20, 't': None}]]}],
}
annotation = {
    'sample_id': 'synthetic-setup', 'device_id': 'synthetic-device',
    'label_status': 'human_verified', 'tokens': {'0': 'x'},
}
checks = 0
for name, valid, mutations in [
    ('aiflow_p_formula_v1.schema.json', record, [
        ('rights_track', 'R'), ('commercial_training_allowed', False),
        ('canvas_width', 0), ('symbols', []), ('split', 'unknown'),
    ]),
    ('aiflow_p_formula_annotation_v1.schema.json', annotation, [
        ('label_status', 'unverified'), ('tokens', {}), ('unexpected', True),
    ]),
]:
    validator = Draft202012Validator(schemas['research/contracts/' + name])
    validator.validate(valid)
    checks += 1
    for key, value in mutations:
        invalid = copy.deepcopy(valid)
        invalid[key] = value
        require(not validator.is_valid(invalid), f'{name} accepted invalid {key}')
        checks += 1
print(f'PASS: {checks} synthetic schema acceptance/rejection checks (no model tests)')

svgs = sorted((ROOT / 'models').rglob('*.svg'))
require(bool(svgs), 'No SVG assets found')
for path in svgs:
    require(ET.parse(path).getroot().tag == '{http://www.w3.org/2000/svg}svg',
            f'Invalid SVG root: {path}')
print(f'PASS: parsed {len(svgs)} SVG assets')

manifest = documents['models/aiflow-math-ink-06-intermediate/MANIFEST.json']
exact = 0
line_endings = []
absent = 0
for entry in manifest['files']:
    source = entry['path']
    candidates = [ROOT / 'models/aiflow-math-ink-06-intermediate' / source]
    if source.startswith(('reports/', 'configs/', 'contracts/')):
        candidates.append(ROOT / 'research' / source)
    local = next((p for p in candidates if p.is_file()), None)
    if local is None:
        absent += 1
        continue
    data = local.read_bytes()
    if len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']:
        exact += 1
        continue
    # The four text report mirrors were converted from CRLF to LF in Git.
    # Report this separately; do not change the manifest or trust model binaries.
    original = data.replace(b'\n', b'\r\n')
    require(source.startswith('reports/') and b'\r' not in data
            and len(original) == entry['bytes']
            and hashlib.sha256(original).hexdigest() == entry['sha256'],
            f'Unexplained manifest mismatch: {local.relative_to(ROOT)}')
    line_endings.append(str(local.relative_to(ROOT)))
require(exact > 0, 'No mirrored manifest entries validated')
print(f'PASS: {exact} exact manifest matches; {len(line_endings)} CRLF-to-LF text mirrors')
for path in line_endings:
    print(f'  LINE ENDING DIFFERENCE: {path}')
print(f'OUT OF SCOPE: {absent} upstream manifest files absent from this documentation mirror')
print('Documentation/data validation complete. No inference, Android build, or upstream test suite ran.')
AIFLOW_VALIDATION_PY
PIP_CACHE_DIR="$setup_dir/pip-cache" "$setup_dir/venv/bin/python" -m pip check
"$setup_dir/venv/bin/python" "$setup_dir/validate.py"

research_env=/workspace/.aiflow-research-env
mkdir -p "$research_env"
python3 -m venv "$research_env/venv"
export PIP_CACHE_DIR="$research_env/pip-cache"
export PYTHONPYCACHEPREFIX="$research_env/pycache"
"$research_env/venv/bin/python" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
cat > "$research_env/requirements.txt" <<'AIFLOW_RESEARCH_PINS'
anyio==4.15.1
brotli==1.2.0
certifi==2026.7.22
click==8.5.0
filelock==3.32.3
fsspec==2026.7.0
gradio_client==2.7.2
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
huggingface_hub==1.33.0
idna==3.20
Jinja2==3.1.6
joblib==1.5.2
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.1.3
orjson==3.12.0
packaging==26.3
pillow==11.3.0
python-dotenv==1.2.4
python-multipart==0.0.32
PyYAML==6.0.3
scikit-learn==1.7.2
scipy==1.15.3
setuptools==78.1.0
starlette==1.7.0
sympy==1.13.1
threadpoolctl==3.7.0
tqdm==4.70.1
trackio==0.40.0
typing_extensions==4.16.0
uvicorn==0.54.0
uvloop==0.23.0
watchfiles==1.3.0
websockets==17.2
AIFLOW_RESEARCH_PINS
"$research_env/venv/bin/python" -m pip install --disable-pip-version-check -r "$research_env/requirements.txt"
"$research_env/venv/bin/python" -m pip check
# Restore only when needed; verify every existing restored file against the manifest.
"$research_env/venv/bin/python" - <<'AIFLOW_ARCHIVE_CHECK'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root = Path('/workspace/AIFlow-Eink')
snapshot = root / 'research/aiflow-1.0e'
manifest = json.loads((snapshot / 'LARGE_ARTIFACTS_MANIFEST.json').read_text())
missing = any(not (root / p).is_file() for r in manifest['blobs'] for p in r['paths'])
if missing:
    subprocess.run([sys.executable, str(snapshot / 'restore_large_artifacts.py')], check=True, cwd=root)
for row in manifest['blobs']:
    for relative in row['paths']:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError('Archive path escapes checkout')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != row['sha256'] or path.stat().st_size != row['bytes']:
            raise RuntimeError(f'Changed archive file preserved: {relative}')
print('PASS: all 39 restored files match archive hashes')
AIFLOW_ARCHIVE_CHECK
cd /workspace/AIFlow-Eink/research/aiflow-1.0e/selective-2d
"$research_env/venv/bin/python" - <<'AIFLOW_RESEARCH_CHECK'
import sys
import torch
sys.path.insert(0, 'scripts')
from cloud_hwr_snapshot import configure
run = configure()
plan, arrays, schedule = run.load()
run.selftest()
_, data, folds, _ = run.direct.broad.previous.global_load(run.direct.broad.previous.OUTPUT)
assert schedule['shared_target_ids'].shape == (2400, 64)
assert len(folds[1]) == len(folds[2]) == 2968
assert torch.__version__ == '2.5.1+cpu'
print('PASS: canonical/data/schedule hashes, near-gap functional tests, two 2968-row development folds')
AIFLOW_RESEARCH_CHECK
