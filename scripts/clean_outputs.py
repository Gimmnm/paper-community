#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理 paper-community 的可再生输出/中间件。
用法示例：
  python scripts/clean_outputs.py          # 仅清理图/输出目录等（安全）
  python scripts/clean_outputs.py --dry-run
  python scripts/clean_outputs.py --all    # 额外清理 data/*.npy 等中间件
  python scripts/clean_outputs.py --all -y # 无交互直接清理
"""
from __future__ import annotations
import argparse, os, sys, glob, shutil

PROJECT_MARKERS = ["pyproject.toml", "README.md", "scripts", "pcore"]

SAFE_PATTERNS = [
    "data/graph/*.csv",         # nodes.csv, edges.csv, communities.csv, layout*.csv 等
    "data/graph/*.gexf",        # 图文件
    "outputs/*",                # 评测/可视化输出
    "logs/*",
    "tmp/*",
]

ALL_EXTRA_PATTERNS = [
    "data/*.npy",               # 例如 embeddings.npy（若从上游生成，建议可删）
    "data/*.npz",
    "data/*.pkl",
    "data/*_layout*.csv",
    "data/*_coords*.csv",
]

def ensure_in_project_root():
    here = os.getcwd()
    ok = any(os.path.exists(os.path.join(here, m)) for m in PROJECT_MARKERS)
    if not ok:
        print("[abort] 当前目录看起来不是项目根目录，请在项目根运行。")
        sys.exit(2)

def collect_targets(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    # 只保留存在的
    files = [p for p in files if os.path.exists(p)]
    # 去重排序
    return sorted(set(files))

def remove_path(path: str):
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            # 既不是文件也不是目录，忽略
            return
        print(f"[del] {path}")
    except Exception as e:
        print(f"[skip] {path}  -> {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="连同 data/*.npy 等中间件一并清理")
    ap.add_argument("--dry-run", action="store_true", help="仅预览将要删除的内容，不实际删除")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过交互确认")
    args = ap.parse_args()

    ensure_in_project_root()

    patterns = list(SAFE_PATTERNS)
    if args.all:
        patterns += ALL_EXTRA_PATTERNS

    targets = collect_targets(patterns)
    if not targets:
        print("[ok] 没有匹配到可删除的文件。")
        return

    print("[plan] 将删除以下路径（共 {} 个）：".format(len(targets)))
    for p in targets:
        print("   -", p)

    if args.dry_run:
        print("[dry-run] 预览模式，未执行删除。")
        return

    if not args.yes:
        resp = input("确认删除？(y/N) ").strip().lower()
        if resp not in ("y", "yes"):
            print("[abort] 已取消。")
            return

    for p in targets:
        remove_path(p)

    print("[done] 清理完成。")

if __name__ == "__main__":
    main()