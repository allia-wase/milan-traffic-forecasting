"""Place the studied squares on the real map of Milan.

Reads the official grid (milano-grid.geojson from the "Milano Grid" Dataverse dataset,
doi:10.7910/DVN/QJWLFU) and writes each square's centre, its distance from the Duomo and a map
link to results/2_exploratory_analysis/square_locations.json. Use the links to check what is
actually in each square before explaining its traffic pattern.

Usage (from the repository root):
    python scripts/03b_locate_squares.py --grid-file <path to milano-grid.geojson>
"""
import argparse
import json
import math
import os
from pathlib import Path

from milan_forecasting import config

DUOMO = (45.46416, 9.19199)  # latitude, longitude of the cathedral square
STUDIED = (5161, 5059, 5259) + config.FOCUS_SQUARES


def cell_id(feature: dict) -> int:
    props = feature["properties"]
    return int(props.get("cellId", props.get("id")))


def centre(feature: dict) -> tuple[float, float]:
    ring = feature["geometry"]["coordinates"][0]
    corners = ring[:-1] if ring[0] == ring[-1] else ring
    return (sum(p[1] for p in corners) / len(corners), sum(p[0] for p in corners) / len(corners))


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def locate(grid: dict, squares) -> dict:
    features = {cell_id(f): f for f in grid["features"]}
    located = {}
    for square in squares:
        lat, lon = centre(features[square])
        located[str(square)] = {
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "km_from_duomo": round(distance_km((lat, lon), DUOMO), 2),
            "map": f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=17/{lat:.5f}/{lon:.5f}",
        }
    return located


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-file", default=os.environ.get("MILAN_GRID_FILE"),
                        help="path to milano-grid.geojson (or set MILAN_GRID_FILE)")
    args = parser.parse_args()
    if not args.grid_file:
        parser.error("pass --grid-file or set MILAN_GRID_FILE")

    grid = json.loads(Path(args.grid_file).read_text(encoding="utf-8"))
    located = locate(grid, STUDIED)
    out = config.ANALYSIS_DIR / "square_locations.json"
    out.write_text(json.dumps(located, indent=2), encoding="utf-8")
    for square, info in located.items():
        print(f"{square}: {info['latitude']}, {info['longitude']}  {info['km_from_duomo']} km from the Duomo  {info['map']}")


if __name__ == "__main__":
    main()
