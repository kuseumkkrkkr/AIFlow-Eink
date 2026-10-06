"""분할 압축된 대용량 연구 파일을 해시 검증하고 원래 경로에 복원한다."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    """메모리 사용을 제한해 파일의 SHA-256을 계산한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Parts(io.RawIOBase):
    """분할 파일을 순서대로 읽으며 각 조각의 크기와 해시를 확인한다."""
    def __init__(self, repo: Path, parts: list[dict]):
        self.repo, self.parts = repo, iter(parts)
        self.stream = None

    def read(self, size: int = -1) -> bytes:
        """tar 스트림이 요청한 크기만 읽어 분할 경계를 연결한다."""
        if size < 0:
            raise ValueError("bounded reads required")
        chunks = []
        while size:
            if self.stream is None:
                self.part = next(self.parts, None)
                if self.part is None:
                    break
                self.stream = destination(self.repo, self.part["path"]).open("rb")
                self.hasher, self.count = hashlib.sha256(), 0
            data = self.stream.read(size)
            if data:
                self.hasher.update(data)
                self.count += len(data)
                chunks.append(data)
                size -= len(data)
            else:
                self.stream.close()
                self.stream = None
                assert self.count == self.part["bytes"], self.part["path"]
                assert self.hasher.hexdigest() == self.part["sha256"], self.part["path"]
        return b"".join(chunks)


def destination(repo: Path, relative: str) -> Path:
    """매니페스트 경로가 저장소 바깥으로 나가지 않게 확인한다."""
    target = (repo / relative).resolve()
    if not target.is_relative_to(repo):
        raise ValueError(f"path outside repository: {relative}")
    return target


def main() -> None:
    """압축 조각과 모든 원본 해시를 확인하고 요청한 경우 파일을 복원한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(__file__).with_name("LARGE_ARTIFACTS_MANIFEST.json").read_text(encoding="utf-8"))
    expected = {row["sha256"]: row for row in manifest["blobs"]}
    reader = Parts(repo, manifest["parts"])
    seen = set()
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for entry in archive:
            key = entry.name.removeprefix("blobs/")
            assert entry.isfile() and entry.name == "blobs/" + key
            assert key in expected and key not in seen
            row = expected[key]
            assert entry.size == row["bytes"]
            targets = [destination(repo, p) for p in row["paths"]]
            missing = []
            if not args.verify_only:
                for target in targets:
                    if target.exists():
                        if sha256(target) != key:
                            raise FileExistsError(f"changed file preserved: {target}")
                    else:
                        missing.append(target)
            temporary = None
            if missing:
                missing[0].parent.mkdir(parents=True, exist_ok=True)
                temporary = missing[0].with_name(missing[0].name + ".restore.tmp")
                sink = temporary.open("xb")
            else:
                sink = None
            h, count = hashlib.sha256(), 0
            try:
                with archive.extractfile(entry) as source:
                    for data in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        h.update(data)
                        count += len(data)
                        if sink:
                            sink.write(data)
            finally:
                if sink:
                    sink.close()
            assert count == row["bytes"] and h.hexdigest() == key, key
            if temporary:
                temporary.replace(missing[0])
                for target in missing[1:]:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(missing[0], target)
                    assert sha256(target) == key, target
            seen.add(key)
            print(f"Verified {len(seen)}/{len(expected)} blobs", flush=True)
    while reader.read(4 * 1024 * 1024):
        pass
    assert seen == set(expected)
    print(json.dumps({"status": "pass", "unique_blobs": len(seen), "original_files": sum(len(r["paths"]) for r in expected.values()), "verify_only": args.verify_only}))


if __name__ == "__main__":
    main()
