"""Convert Kaggle's TMDB 5000 CSV files to CineSense's compact catalog.

Usage: python scripts/build_tmdb_catalog.py path/to/tmdb-movie-metadata.zip
"""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "movies_tmdb.csv"
FIELDS = ("movie_id", "title", "year", "genres", "keywords", "cast", "director", "overview")


def names(value: str) -> list[str]:
    """Read the name fields from a TMDB JSON list."""
    return [item["name"] for item in json.loads(value or "[]") if item.get("name")]


def build_catalog(archive_path: Path, output_path: Path = OUTPUT) -> int:
    """Join credits to movies by ID, then keep only recommendation fields."""
    with ZipFile(archive_path) as archive:
        with archive.open("tmdb_5000_credits.csv") as raw:
            credits = {
                row["movie_id"]: row
                for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
            }
        with archive.open("tmdb_5000_movies.csv") as raw:
            movies = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))

    seen_ids = set()
    records = []
    for movie in movies:
        movie_id = movie["id"].strip()
        title = movie["title"].strip()
        if not movie_id or not title or movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)
        credit = credits.get(movie_id, {})
        crew = json.loads(credit.get("crew") or "[]")
        directors = [person["name"] for person in crew if person.get("job") == "Director"]
        records.append({
            "movie_id": f"tmdb:{movie_id}",
            "title": title,
            "year": movie["release_date"][:4],
            "genres": "|".join(names(movie["genres"])),
            "keywords": "|".join(names(movie["keywords"])),
            "cast": "|".join(names(credit.get("cast") or "[]")[:5]),
            "director": "|".join(directors),
            "overview": movie["overview"].strip(),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="TMDB 5000 zip downloaded from Kaggle")
    args = parser.parse_args()
    print(f"Wrote {build_catalog(args.archive)} movies to {OUTPUT}")
