"""Convert the Kaggle TMDB archive into the CineSense catalog format."""

import argparse
import csv
import gzip
import heapq
import io
from pathlib import Path
import zipfile


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "movies_tmdb_kaggle.csv.gz"
FIELDS = ("movie_id", "title", "year", "genres", "keywords", "cast", "director", "overview")


def import_catalog(archive: Path, *, output: Path = OUTPUT, max_movies: int | None = None) -> int:
    """Import all rows, or the most-voted movies with genres and overviews."""
    if max_movies is not None and max_movies < 1:
        raise ValueError("max_movies must be positive.")
    seen_ids = set()
    temporary = output.with_suffix(".tmp")
    try:
        with zipfile.ZipFile(archive) as source:
            csv_files = [name for name in source.namelist() if name.lower().endswith(".csv")]
            if len(csv_files) != 1:
                raise ValueError("Expected a Kaggle archive containing one CSV.")
            with source.open(csv_files[0]) as raw, gzip.open(
                temporary, "wt", encoding="utf-8", newline="", compresslevel=3
            ) as target:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                required = {"id", "title", "release_date", "genres", "keywords", "overview"}
                if max_movies is not None:
                    required.add("vote_count")
                if not required.issubset(rows.fieldnames or ()):
                    raise ValueError("The Kaggle CSV is missing required movie metadata.")
                def usable_rows():
                    for row in rows:
                        movie_id = (row["id"] or "").strip()
                        title = (row["title"] or "").strip()
                        if not movie_id.isdecimal() or not title or movie_id in seen_ids:
                            continue
                        if max_movies is not None and not (
                            (row["genres"] or "").strip() and (row["overview"] or "").strip()
                        ):
                            continue
                        seen_ids.add(movie_id)
                        yield row

                selected = usable_rows()
                if max_movies is not None:
                    def rank(row):
                        votes = (row["vote_count"] or "").strip()
                        return (int(votes) if votes.isdecimal() else 0, -int(row["id"]))
                    selected = heapq.nlargest(max_movies, selected, key=rank)
                writer = csv.writer(target)
                writer.writerow(FIELDS)
                count = 0
                for row in selected:
                    year = (row["release_date"] or "")[:4]
                    writer.writerow((
                        f"tmdb:{row['id'].strip()}", row["title"].strip(),
                        year if year.isdecimal() else "",
                        (row["genres"] or "").replace(", ", "|"),
                        (row["keywords"] or "").replace(", ", "|"),
                        "", "", row["overview"] or "",
                    ))
                    count += 1
        if not count:
            raise ValueError("The Kaggle CSV contains no usable movies.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded Kaggle dataset ZIP")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-movies", type=int, help="Keep the most-voted movies with genres and overviews")
    args = parser.parse_args()
    count = import_catalog(args.archive, output=args.output, max_movies=args.max_movies)
    print(f"Imported {count:,} movies into {args.output}")
