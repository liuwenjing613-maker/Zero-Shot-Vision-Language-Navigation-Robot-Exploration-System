"""Run the controlled M1 checks, then the real MP3D checks."""
import argparse
from pathlib import Path

import controlled

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--mode", choices=("controlled", "mp3d", "both", "scans"), default="both"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/chenkejun/datasets/mp3d_habitat/scenes"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "/data/chenkejun/vln/vln-perception/results/mp3d-2azQ1b91cZZ-chair-challenges-v1/multiple.png"
        ),
    )
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    if args.mode in ("controlled", "both"):
        controlled.run(args.folder / "controlled")
    if args.mode in ("mp3d", "both"):
        import mp3d

        mp3d.run(args.folder / "mp3d", args.dataset, args.archive, args.gpu)
    if args.mode == "scans":
        import scan_lab

        scan_lab.room(args.folder / "room360")
        import scan_mp3d

        scan_mp3d.run(args.folder / "laser360", args.dataset, args.gpu)
