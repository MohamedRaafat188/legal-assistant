"""Dump every amended article from chunks_law72.json into a plain-text file
for human review -- the JSON escaping makes them painful to read in place."""
import json
from pathlib import Path

SP = Path(__file__).parent
chunks = json.loads((SP / "chunks_law72.json").read_text(encoding="utf-8"))
amended = [c for c in chunks if c["metadata"]["amendments"]]

lines = [
    f"Law 72/2017 -- amended articles for review ({len(amended)} of {len(chunks)} chunks)",
    "=" * 70,
    "",
]
for c in amended:
    lines.append(f"[{c['chunk_id']}]")
    lines.append("-" * 70)
    lines.append(c["text_for_display"])
    lines.append("")
    lines.append("amendments metadata:")
    for a in c["metadata"]["amendments"]:
        lines.append(f"  - {a}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

out = SP / "law72_amended_articles.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out} ({len(amended)} articles)")
