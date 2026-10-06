"""Use byte-identical archived caches without editing historical research scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT.parent


def configure():
    """Replace only two machine-specific input locations; retain all SHA checks."""
    import run_hwr_near_rival_training_v33 as run

    previous = run.direct.broad.previous
    checkpoint = SNAPSHOT / 'augmentation-models/canonical/project_symbol_head_checkpoint.pt'
    previous.CHECKPOINT = checkpoint
    previous.DATA = SNAPSHOT / 'augmentation-models/affine-distill-v1'
    return run
