from __future__ import annotations

import argparse
from pathlib import Path


REPO_ID = "nvidia/HiLiftAeroML"
DEFAULT_GEOMETRIES = tuple(range(1, 5))
DEFAULT_AOAS = (4, 8, 12, 16, 20)
ROOT_FILES = ("force_mom_all.csv", "geo_values_all.csv")
SPLIT_PREFIX = "splits/"


def parse_int_set(value: str) -> list[int]:
    """Parse comma-separated IDs with optional ranges, e.g. 1,4-6,10."""
    items: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = [int(part.strip()) for part in chunk.split("-", 1)]
            if hi < lo:
                raise ValueError(f"Invalid descending range: {chunk}")
            items.update(range(lo, hi + 1))
        else:
            items.add(int(chunk))
    return sorted(items)


def run_id(geometry: int, aoa: int) -> str:
    return f"geo_LHC{geometry:03d}_AoA_{aoa}"


def selected_run_files(
    geometries: list[int],
    aoas: list[int],
    *,
    include_images: bool,
    include_stl: bool,
    include_step: bool,
    include_boundary: bool,
    include_volume: bool,
) -> list[str]:
    files: list[str] = []
    for geometry in geometries:
        if not 1 <= geometry <= 180:
            raise ValueError(f"Geometry ID must be in [1, 180], got {geometry}")
        for aoa in aoas:
            if aoa not in range(4, 23, 2):
                raise ValueError(f"AoA must be one of 4, 6, ..., 22, got {aoa}")
            rid = run_id(geometry, aoa)
            files.extend(
                [
                    f"{rid}/force_mom_{rid}.csv",
                    f"{rid}/geo_values_{rid}.csv",
                    f"{rid}/ref_values_{rid}.csv",
                ]
            )
            if include_images:
                files.extend(
                    [
                        f"{rid}/img_wss_LHC{geometry:03d}_AoA_{aoa}.png",
                        f"{rid}/plot_CD_{rid}.png",
                        f"{rid}/plot_CL_{rid}.png",
                        f"{rid}/plot_CM_{rid}.png",
                    ]
                )
            if include_stl:
                files.append(f"{rid}/{rid}.stl")
            if include_step:
                files.append(f"{rid}/{rid}.stp")
            if include_boundary:
                files.append(f"{rid}/boundary_{rid}.vtu.tgz")
            if include_volume:
                files.append(f"{rid}/volume_{rid}.vtu.tgz")
    return files


def build_file_list(all_repo_files: list[str], args: argparse.Namespace) -> list[str]:
    geometries = parse_int_set(args.geometries)
    aoas = parse_int_set(args.aoas)
    selected = set(ROOT_FILES)
    selected.update(path for path in all_repo_files if path.startswith(SPLIT_PREFIX))
    selected.update(
        selected_run_files(
            geometries,
            aoas,
            include_images=args.include_images,
            include_stl=args.include_stl,
            include_step=args.include_step,
            include_boundary=args.include_boundary,
            include_volume=args.include_volume,
        )
    )
    available = set(all_repo_files)
    missing = sorted(path for path in selected if path not in available)
    if missing:
        preview = "\n  ".join(missing[:12])
        raise RuntimeError(f"{len(missing)} selected files were not found in {REPO_ID}:\n  {preview}")
    return sorted(selected)


def print_plan(files: list[str], args: argparse.Namespace) -> None:
    n_runs = len(parse_int_set(args.geometries)) * len(parse_int_set(args.aoas))
    print(f"repo: {REPO_ID}")
    print(f"local_dir: {Path(args.local_dir).resolve()}")
    print(f"selected runs: {n_runs}")
    print(f"selected files: {len(files)}")
    print()
    print("This is the small real-data path: CSV summaries, per-run coefficients/reference values, and splits.")
    if args.include_images:
        print("Also includes lightweight rendered WSS/convergence PNGs for selected runs.")
    if args.include_stl or args.include_step or args.include_boundary or args.include_volume:
        print()
        print("Large-file options enabled. Dataset card approximate compressed sizes per run:")
        if args.include_stl:
            print("- STL geometry: ~197 MB")
        if args.include_step:
            print("- STP geometry: ~48 MB")
        if args.include_boundary:
            print("- boundary surface fields: ~13 GB compressed")
        if args.include_volume:
            print("- volume fields: ~23 GB compressed")
    print()
    for path in files[:40]:
        print(path)
    if len(files) > 40:
        print(f"... {len(files) - 40} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small, explicit real-data subset from nvidia/HiLiftAeroML."
    )
    parser.add_argument("--local-dir", default="data/hiliftaeroml_subset")
    parser.add_argument(
        "--geometries",
        default=",".join(str(x) for x in DEFAULT_GEOMETRIES),
        help="Geometry IDs, comma-separated with ranges allowed, e.g. 1-4,12.",
    )
    parser.add_argument(
        "--aoas",
        default=",".join(str(x) for x in DEFAULT_AOAS),
        help="AoA values, comma-separated with ranges allowed. Valid values: 4,6,...,22.",
    )
    parser.add_argument("--include-images", action="store_true", help="Include WSS and coefficient plot PNGs.")
    parser.add_argument("--include-stl", action="store_true", help="Include selected STL geometry files.")
    parser.add_argument("--include-step", action="store_true", help="Include selected STP geometry files.")
    parser.add_argument("--include-boundary", action="store_true", help="Include selected boundary_*.vtu.tgz files.")
    parser.add_argument("--include-volume", action="store_true", help="Include selected volume_*.vtu.tgz files.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download. Without this flag the script only previews the file list.",
    )
    parser.add_argument(
        "--i-understand-large-download",
        action="store_true",
        help="Required with STL/STP/boundary/volume options.",
    )
    args = parser.parse_args()

    large_requested = args.include_stl or args.include_step or args.include_boundary or args.include_volume
    if large_requested and not args.i_understand_large_download:
        raise SystemExit("Large-file options require --i-understand-large-download.")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install dependencies with: uv sync --extra realdata") from exc

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    files = build_file_list(all_files, args)
    print_plan(files, args)
    if not args.execute:
        print()
        print("Preview only. Add --execute to download this subset.")
        return

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(f"downloading {path}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=path,
            local_dir=local_dir,
        )
    print(f"done: {local_dir.resolve()}")


if __name__ == "__main__":
    main()
