"""기록된 코드 SHA와 일치하는 연구 소스를 새 산출물 폴더에 보존한다."""
import argparse
import ast
import json
import shutil
from pathlib import Path
from accuracy_upgrade_contract_v1 import sha256_file
from accuracy_upgrade_data_10e import write_json


def snapshot(experiment: Path):
    """실행 당시 manifest와 대조한 소스 및 local import 의존성을 복사한다."""
    root=Path(__file__).resolve().parent
    manifest=json.loads((experiment/'model_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['code_sha256'].items():
        if sha256_file(str(root/name)) != expected:
            raise ValueError('cannot reconstruct changed execution source: '+name)
    output=experiment/'source_snapshot'
    if output.exists():
        raise FileExistsError(output)
    pending=[root/name for name in manifest['code_sha256']]
    pending += [root/'run_accuracy_hwr_10e.py',root/'accuracy_hwr_finetune_10e.py']
    found={}
    while pending:
        path=pending.pop()
        if path.name in found:
            continue
        found[path.name]=sha256_file(str(path))
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                   [x.name for x in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                local=root/(name.split('.')[0]+'.py')
                if local.exists() and local.name not in found:
                    pending.append(local)
    output.mkdir()
    for name,expected in found.items():
        shutil.copy2(root/name,output/name)
        if sha256_file(str(output/name)) != expected:
            raise ValueError('code changed during snapshot: '+name)
    write_json(output/'manifest.json',dict(files=found,scope='research reproducibility snapshot, not a product package'))
    return dict(experiment=str(experiment),files=len(found))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiments',type=Path,nargs='+');a=p.parse_args()
    for experiment in a.experiments:
        print(json.dumps(snapshot(experiment)),flush=True)
