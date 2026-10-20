"""Write a coverage badge SVG from coverage.xml (CI publishes it to the `badges` branch).

Usage: python scripts/coverage_badge.py coverage.xml badges/coverage.svg
"""
import sys
import xml.etree.ElementTree as ET  # noqa: S405 - our own coverage.xml
from pathlib import Path

TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="104" height="20" role="img" aria-label="coverage: {pct}">
<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
<stop offset="1" stop-opacity=".1"/></linearGradient>
<rect rx="3" width="104" height="20" fill="#555"/><rect rx="3" x="61" width="43" height="20" fill="{color}"/>
<rect rx="3" width="104" height="20" fill="url(#s)"/>
<g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">
<text x="31" y="14">coverage</text><text x="81.5" y="14">{pct}</text></g></svg>
"""


def main() -> None:
    rate = float(ET.parse(sys.argv[1]).getroot().get("line-rate"))  # noqa: S314
    pct = f"{rate:.0%}"
    color = "#4c1" if rate >= 0.9 else "#dfb317" if rate >= 0.75 else "#e05d44"
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.format(pct=pct, color=color), encoding="utf-8")
    print(pct)


if __name__ == "__main__":
    main()
