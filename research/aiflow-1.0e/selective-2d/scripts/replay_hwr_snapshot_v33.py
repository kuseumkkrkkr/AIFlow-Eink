"""Re-run the unchanged V33 verifier in a separate output directory.

Historical certificates, data, checkpoints, scripts and hashes remain unchanged.
No private raw participant files, remote services or optimizer steps are needed.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil

from cloud_hwr_snapshot import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f'Replay output already exists: {target}')
    run = configure()
    source = run.OUT
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Replay must be separate from the historical experiment')
    import verify_hwr_near_rival_training_v33 as verifier
    import numpy as np
    import torch

    shutil.copytree(source, target, ignore=shutil.ignore_patterns('independent_verification.json', 'trackio'))
    run.OUT = target
    status = {'status': 'interrupted'}
    try:
        verifier.main()
    except Exception as exc:
        status = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
        raise
    else:
        status = {'status': 'pass', 'historical_verifier_unchanged': True}
    finally:
        metadata = dict(status, python=platform.python_version(), platform=platform.platform(),
                        torch=torch.__version__, numpy=np.__version__,
                        historical_source=str(source), optimizer_steps=0)
        (target / 'cloud_replay_status.json').write_text(json.dumps(metadata, indent=2) + '\n')
        run.OUT = source


if __name__ == '__main__':
    main()
