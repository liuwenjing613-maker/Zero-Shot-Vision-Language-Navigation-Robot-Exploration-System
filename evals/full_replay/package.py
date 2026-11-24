"""Produce one offline HTML file containing the verified replay assets."""
from pathlib import Path
import json
from options import SOURCE, options

args = options()
html = (SOURCE / "layout.html").read_text()
for marker, file in [
    ("STYLES", "style.css"),
    ("APP", "app.js"),
    ("RECALL_JS", "recall.js"),
]:
    assert html.count("/*" + marker + "*/") == 1
    html = html.replace("/*" + marker + "*/", (SOURCE / file).read_text())
for marker, file in [
    ("AREAS", "areas.json"),
    ("MOTION", "motion.json"),
    ("DATA", "dataset.json"),
    ("BOUNDS", "bounds.json"),
    ("METRICS", "motion-checks.json"),
]:
    assert html.count("/*" + marker + "*/") == 1
    data = (args.output / file).read_text()
    html = html.replace("/*" + marker + "*/", data.replace("<", "\\u003c"))
report = args.memory_trial.read_text() if args.memory_trial else "null"
html = html.replace("/*RECALL_DATA*/", report.replace("<", r"\u003c"))
final = args.output / "index.html"
staged = final.with_suffix(".html.tmp")
staged.write_text(html)
staged.replace(final)
print(final, flush=True)
