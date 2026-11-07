"""Keep independently deployable Spaces in sync without runtime parent imports.

The English main Space owns the shared JSON, SVG and main requirements. The
English VVC demo owns requirements shared by the ten environment demos.
"""

import argparse
import shutil
from pathlib import Path


DEMO_SPACES = tuple(
    f"powerzoo-{family}{suffix}"
    for family in ("vvc", "smartgrid", "dsr", "stackelberg", "district-dispatch")
    for suffix in ("", "-zh")
)


def synchronize(root: Path, *, check: bool = True) -> list[str]:
    """Return drift descriptions, or repair derived files when check is false."""
    spaces = root / "huggingface_spaces"
    source = spaces / "powerzoo"
    translated = spaces / "powerzoo-zh"
    source_data = source / "data"
    if not source_data.is_dir() or not any(source_data.glob("*.json")):
        raise FileNotFoundError(f"Missing canonical JSON data: {source_data}")
    pairs = [(path, translated / "data" / path.name)
             for path in sorted(source_data.glob("*.json"))]
    pairs.extend((source / name, translated / name)
                 for name in ("requirements.txt", "assets/architecture.svg"))
    demo_requirements = spaces / "powerzoo-vvc" / "requirements.txt"
    pairs.extend((demo_requirements, spaces / name / "requirements.txt")
                 for name in DEMO_SPACES if name != "powerzoo-vvc")
    # Validate every canonical file before writing any destination.
    for original, _ in pairs:
        if not original.is_file():
            raise FileNotFoundError(f"Missing canonical resource: {original}")
    drift = []
    for original, derived in pairs:
        if not derived.is_file() or original.read_bytes() != derived.read_bytes():
            drift.append(str(derived.relative_to(root)))
            if not check:
                derived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, derived)
    canonical_names = {path.name for path in source_data.glob("*.json")}
    for stale in sorted((translated / "data").glob("*.json")):
        if stale.name not in canonical_names:
            drift.append(f"stale: {stale.relative_to(root)}")
            if not check:
                stale.unlink()
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail if shared resources differ")
    mode.add_argument("--write", action="store_true", help="Refresh derived copies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    drift = synchronize(root, check=not args.write)
    for path in drift:
        print(path)
    if args.check and drift:
        print("Shared resources differ; run python tools/sync_hf_shared_assets.py --write")
        return 1
    print(f"Shared Space resources {'updated' if args.write else 'consistent'} ({len(drift)} differences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
