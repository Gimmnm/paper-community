from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List


DEFAULT_CANDIDATE_PREFIXES = (
    "leiden_hierarchy",
    "subgraph_hierarchy",
    "coarse_domains",
    "topic_viz",
)


def list_candidate_dirs(out_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "experiments":
            continue
        if p.name.startswith(DEFAULT_CANDIDATE_PREFIXES):
            out.append(p)
    return out


def archive_candidates(base_dir: Path, execute: bool = False) -> List[str]:
    out_dir = base_dir / "out"
    archive_root = base_dir / "archive" / "legacy" / "out"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    moved: List[str] = []

    for src in list_candidate_dirs(out_dir):
        dst = archive_root / f"{src.name}__{stamp}"
        if execute:
            if dst.exists():
                raise FileExistsError(f"target exists: {dst}")
            shutil.move(str(src), str(dst))
        moved.append(f"{src} -> {dst}")
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive deprecated out/ directories by naming convention.")
    ap.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--execute", action="store_true", help="actually move folders; default is dry-run")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    rows = archive_candidates(base_dir=base, execute=bool(args.execute))
    if not rows:
        print("[archive-out] no candidate directories found")
        return
    print("[archive-out] candidates:")
    for x in rows:
        print(f"  - {x}")
    if not args.execute:
        print("[archive-out] dry-run only. use --execute to move.")


if __name__ == "__main__":
    main()
