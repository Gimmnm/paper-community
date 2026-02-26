import argparse
import re
from pathlib import Path
from typing import Iterable
import numpy as np  # 修复点：使用 numpy.asarray，而不是 imageio.asarray


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def list_frames(frame_dir: Path, exts=(".png", ".jpg", ".jpeg", ".webp")):
    files = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def _import_backends():
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 imageio。请安装: pip install imageio imageio-ffmpeg pillow"
        ) from e
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 Pillow。请安装: pip install pillow"
        ) from e
    return imageio, Image


def make_mp4(
    frame_dir: Path,
    out_path: Path | None = None,
    fps: int = 8,
    pattern: str | None = None,
    recursive: bool = False,
    resize_mode: str = "first",  # first | even
    quality: int | None = None,
    codec: str = "libx264",
    verbose: bool = True,
):
    imageio, Image = _import_backends()

    if not frame_dir.exists() or not frame_dir.is_dir():
        raise FileNotFoundError(f"帧目录不存在: {frame_dir}")

    frames = list_frames(frame_dir)
    if pattern:
        rx = re.compile(pattern)
        frames = [p for p in frames if rx.search(p.name)]
    if not frames and recursive:
        # 递归收集时，按“每个子目录一条视频”在 batch 模式里处理；这里保持单目录逻辑简单
        raise RuntimeError("当前模式不支持 recursive=True 的单目录合成，请使用 batch 模式（--batch-subdirs）")
    if not frames:
        raise RuntimeError(f"未找到帧图片: {frame_dir}")

    if out_path is None:
        out_path = frame_dir.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取首帧确定尺寸
    with Image.open(frames[0]) as im0:
        w0, h0 = im0.size
    if resize_mode == "even":
        # 编码器常要求偶数尺寸
        target_w = w0 if w0 % 2 == 0 else w0 - 1
        target_h = h0 if h0 % 2 == 0 else h0 - 1
    else:
        target_w, target_h = w0, h0
        if target_w % 2 == 1:
            target_w -= 1
        if target_h % 2 == 1:
            target_h -= 1

    if verbose:
        print(f"[mp4] frame_dir={frame_dir}")
        print(f"[mp4] frames={len(frames)}, fps={fps}, out={out_path}")
        print(f"[mp4] target_size={target_w}x{target_h}, codec={codec}")

    writer_kwargs = {
        "fps": fps,
        "codec": codec,
    }
    # imageio ffmpeg 常用参数：quality 越小质量越高（但不同版本处理略有差异）
    if quality is not None:
        writer_kwargs["quality"] = int(quality)

    # Pillow 新旧版本兼容（不影响你路径和文件名）
    try:
        resample_bicubic = Image.Resampling.BICUBIC  # Pillow >= 9/10
    except AttributeError:
        resample_bicubic = Image.BICUBIC  # 旧版 Pillow

    with imageio.get_writer(str(out_path), format="ffmpeg", **writer_kwargs) as writer:
        for i, fp in enumerate(frames, start=1):
            with Image.open(fp) as im:
                im = im.convert("RGB")
                if im.size != (target_w, target_h):
                    im = im.resize((target_w, target_h), resample=resample_bicubic)

                # 关键修复：不要用 imageio.asarray(im)
                writer.append_data(np.asarray(im))

            if verbose and (i == 1 or i == len(frames) or i % 50 == 0):
                print(f"[mp4] appended {i}/{len(frames)}: {fp.name}")

    if verbose:
        print(f"[mp4] saved -> {out_path}")
    return out_path


def batch_make_mp4s(
    root: Path,
    subdir_glob: str = "frames*",
    fps: int = 8,
    pattern: str | None = None,
    out_root: Path | None = None,
    resize_mode: str = "first",
    quality: int | None = None,
    codec: str = "libx264",
    verbose: bool = True,
):
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"根目录不存在: {root}")
    dirs = [p for p in root.iterdir() if p.is_dir() and p.match(subdir_glob)]
    dirs.sort(key=lambda p: natural_key(p.name))
    if not dirs:
        raise RuntimeError(f"在 {root} 下未找到匹配子目录: {subdir_glob}")

    outputs = []
    for d in dirs:
        out_path = (out_root / f"{d.name}.mp4") if out_root else (root / f"{d.name}.mp4")
        try:
            op = make_mp4(
                d,
                out_path=out_path,
                fps=fps,
                pattern=pattern,
                recursive=False,
                resize_mode=resize_mode,
                quality=quality,
                codec=codec,
                verbose=verbose,
            )
            outputs.append(op)
        except Exception as e:
            print(f"[mp4] skip {d}: {e}")
    return outputs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="将帧图合并为 MP4（单目录或批量子目录）")
    p.add_argument("--frame-dir", type=str, default=None, help="单个帧目录，例如 out/topic_viz_multi/K10/frames_topic_umap")
    p.add_argument("--out", type=str, default=None, help="单目录模式输出 MP4 路径")
    p.add_argument("--fps", type=int, default=8, help="帧率")
    p.add_argument("--pattern", type=str, default=None, help=r"文件名正则筛选，例如 r'\\.png$' 或 r'r1\\.0000'")
    p.add_argument("--resize-mode", type=str, choices=["first", "even"], default="first")
    p.add_argument("--codec", type=str, default="libx264")
    p.add_argument("--quality", type=int, default=None)

    p.add_argument("--batch-subdirs", action="store_true", help="批量模式：把 root 下匹配的子目录各自合成一个 MP4")
    p.add_argument("--root", type=str, default=None, help="批量模式根目录，例如 out/topic_viz_multi/K10")
    p.add_argument("--subdir-glob", type=str, default="frames*", help="批量模式子目录匹配模式")
    p.add_argument("--out-root", type=str, default=None, help="批量模式输出目录（默认写到 root）")

    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    verbose = not args.quiet

    if args.batch_subdirs:
        if not args.root:
            raise SystemExit("批量模式需要 --root")
        outs = batch_make_mp4s(
            root=Path(args.root),
            subdir_glob=args.subdir_glob,
            fps=args.fps,
            pattern=args.pattern,
            out_root=Path(args.out_root) if args.out_root else None,
            resize_mode=args.resize_mode,
            quality=args.quality,
            codec=args.codec,
            verbose=verbose,
        )
        print(f"[mp4] batch done. videos={len(outs)}")
        for p in outs:
            print(f"[mp4] {p}")
        return

    if not args.frame_dir:
        raise SystemExit("单目录模式需要 --frame-dir；或使用 --batch-subdirs --root")

    make_mp4(
        frame_dir=Path(args.frame_dir),
        out_path=Path(args.out) if args.out else None,
        fps=args.fps,
        pattern=args.pattern,
        resize_mode=args.resize_mode,
        quality=args.quality,
        codec=args.codec,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()