"""Build a clean group handoff ZIP, excluding environments and course data."""

import hashlib
import json
from pathlib import Path
import zipfile

from .stage1_data import ROOT


def main():
    checks = ROOT / "outputs/stage1/demand_checks.json"
    if not checks.exists() or json.loads(checks.read_text())["status"] != "passed":
        raise RuntimeError("Run the handoff checks before packaging")
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if (len(rel.parts) == 1 and path.suffix in [".md", ".txt", ".json"]) or (
            rel.parts[0] in ["experiments", "harness"] and path.suffix in [".py", ".json"]
        ) or rel.parts[0] in ["outputs", "artifacts", "handoffs"]:
            files.append(path)
    destination = ROOT.parent / "A_demand_handoff.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, ROOT.name + "/" + str(path.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP checksum verification failed")
        for path in files:
            name = ROOT.name + "/" + str(path.relative_to(ROOT))
            if archive.read(name) != path.read_bytes():
                raise RuntimeError(f"ZIP differs: {name}")
    print(f"Packed {len(files)} files: {destination} ({destination.stat().st_size / 1024**2:.2f} MiB)")
    print("SHA256", hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
