#!/usr/bin/env python3
"""
archive から NA_Fish_Dataset ディレクトリだけを抽出してデータセット用ディレクトリに展開するスクリプト。

使用例:
    python dataset --archive /path/to/archive.zip --output ./dataset
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable


def _validate_member_path(member: str) -> None:
    """メンバーのパスが危険な値（絶対パスや .. を含む）でないかを検証する。"""
    path = Path(member)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"危険なパスを検出したため処理を中止しました: {member}")


def _iter_dataset_members_zip(zf: zipfile.ZipFile, dataset_name: str) -> Iterable[str]:
    prefix = f"{dataset_name.rstrip('/')}/"
    for member in zf.namelist():
        if member == dataset_name or member.startswith(prefix):
            yield member


def _iter_dataset_members_tar(tf: tarfile.TarFile, dataset_name: str) -> Iterable[tarfile.TarInfo]:
    prefix = f"{dataset_name.rstrip('/')}/"
    for member in tf.getmembers():
        name = member.name
        if name == dataset_name or name.startswith(prefix):
            yield member


def extract_na_fish_dataset(archive_path: Path, output_dir: Path, dataset_name: str = "NA_Fish_Dataset") -> Path:
    """
    指定したアーカイブから dataset_name ディレクトリのみを抽出して出力先に展開する。

    Parameters
    ----------
    archive_path : Path
        抽出元アーカイブへのパス（.zip, .tar, .tar.gz, .tar.bz2, .tar.xz をサポート）。
    output_dir : Path
        展開先ディレクトリ。
    dataset_name : str, optional
        抽出対象ディレクトリ名（デフォルト: "NA_Fish_Dataset"）。

    Returns
    -------
    Path
        展開後のデータセットディレクトリへのパス。
    """
    archive_path = archive_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not archive_path.exists():
        raise FileNotFoundError(f"アーカイブが見つかりません: {archive_path}")

    target_root = output_dir / dataset_name
    if target_root.exists():
        shutil.rmtree(target_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive_path.suffixes).lower()

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            members = list(_iter_dataset_members_zip(zf, dataset_name))
            if not members:
                raise ValueError(f"アーカイブ内に {dataset_name} が見つかりませんでした。")
            for member in members:
                _validate_member_path(member)
                destination = output_dir / member
                if member.endswith("/"):
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    elif suffixes in {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz"}:
        mode = "r:gz" if suffixes in {".tar.gz", ".tgz"} else \
               "r:bz2" if suffixes in {".tar.bz2", ".tbz"} else \
               "r:xz" if suffixes in {".tar.xz", ".txz"} else "r:"
        with tarfile.open(archive_path, mode) as tf:
            members = list(_iter_dataset_members_tar(tf, dataset_name))
            if not members:
                raise ValueError(f"アーカイブ内に {dataset_name} が見つかりませんでした。")
            for member in members:
                _validate_member_path(member.name)
                tf.extract(member, path=output_dir)
    else:
        raise ValueError(f"未対応のアーカイブ形式です: {archive_path}")

    if not target_root.exists():
        raise RuntimeError(f"抽出後に {target_root} が存在しません。")

    return target_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="archive から NA_Fish_Dataset のみを抽出してデータセットを構築します。"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("./archive.zip"),
        help="抽出元アーカイブのパス（デフォルト: ./archive.zip）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./dataset"),
        help="データセット展開先ディレクトリ（デフォルト: ./dataset）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="NA_Fish_Dataset",
        help="抽出対象ディレクトリ名（デフォルト: NA_Fish_Dataset）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = extract_na_fish_dataset(args.archive, args.output, args.name)
    print(f"{dataset_path} に NA_Fish_Dataset を展開しました。")


if __name__ == "__main__":
    main()

