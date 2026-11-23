"""Embed measured frames and per-step results for the existing SSH-served M1 lab."""
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from PIL import Image


def photo(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def package(root, out):
    cases = {}
    for k in json.loads((root / "routes.json").read_text()):
        folder = root / k
        d = json.loads((folder / "record.json").read_text())
        cases[k] = d
        buffer = BytesIO()
        with Image.open(folder / "topdown.jpg") as img:
            img.resize((800, 800)).save(buffer, format="JPEG", quality=84)
        first = root / "first" / (k + ".json")
        d["first_missed"] = first.is_file() and not all(
            json.loads(first.read_text())["checks"].values()
        )
        d["reference"] = (
            "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        )
        d["images"] = [photo(folder / f"rgb-{r['t']}.jpg") for r in d["trace"]]
        d["maps"] = [
            "data:image/png;base64,"
            + base64.b64encode((folder / f"grid-{r['t']}.png").read_bytes()).decode()
            for r in d["trace"]
        ]
    out.mkdir(exist_ok=True, parents=True)
    source = Path(__file__).with_name("replay.html").read_text()
    (out / "index.html").write_text(
        source.replace(
            "__JUNCTION_DATA__",
            json.dumps(cases, separators=(",", ":")).replace("<", "\\u003c"),
        )
    )


if __name__ == "__main__":
    package(Path(sys.argv[1]), Path(sys.argv[2]))
