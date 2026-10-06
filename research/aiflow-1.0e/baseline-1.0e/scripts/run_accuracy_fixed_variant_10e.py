"""승인된 시간 채널·재현 seed만 바꿔 기존 실행기를 호출한다."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from accuracy_upgrade_data_10e import write_json
from accuracy_upgrade_contract_v1 import sha256_file


def main():
    """부모 설정 SHA와 변경 축을 새 설정 파일에 고정하여 실행한다."""
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--variant-config',type=Path,required=True)
    p.add_argument('--input-mode',choices=['preserve','uniform-time'])
    p.add_argument('--seed',type=int,choices=[20260910,20260911,20260912])
    p.add_argument('--runner',choices=['upgrade','hwr'],default='upgrade')
    a, remaining=p.parse_known_args()
    if a.variant_config.exists() or a.variant_config.resolve().drive.upper()!='D:':
        raise ValueError('new D: variant config required')
    config=json.loads(a.config.read_text(encoding='utf-8'))
    changes={}
    if a.input_mode is not None:
        changes['input_mode']=a.input_mode
    if a.seed is not None:
        changes['seed']=a.seed
    if not changes:
        raise ValueError('variant needs one declared axis')
    config.update(changes)
    config['variant_provenance']=dict(parent=str(a.config.resolve()),parent_sha256=sha256_file(str(a.config)),changes=changes)
    write_json(a.variant_config,config)
    filename='run_accuracy_hwr_10e.py' if a.runner=='hwr' else 'run_accuracy_upgrade_10e.py'
    command=[sys.executable,'-s','-B',str(Path(__file__).with_name(filename)),'--config',str(a.variant_config),*remaining]
    return subprocess.call(command)


if __name__=='__main__':
    raise SystemExit(main())
