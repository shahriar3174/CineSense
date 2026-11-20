"""Hosted catalog generation and recommendation integration."""

import csv
import io
import zipfile

import pandas as pd
import pytest

import poc_recommender as recommender
from scripts.import_kaggle_catalog import import_catalog


def archive_with_rows(tmp_path, rows):
    content = io.StringIO()
    writer = csv.writer(content)
    writer.writerow(["id", "title", "release_date", "genres", "keywords", "overview", "vote_count"])
    writer.writerows(rows)
    archive = tmp_path / "movies.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("movies.csv", content.getvalue())
    return archive


def test_subset_selects_top_votes_with_stable_ties_and_usable_metadata(tmp_path):
    archive = archive_with_rows(tmp_path, [
        ["3", "Third", "2003-01-01", "Action, Drama", "robot", "robot story", "20"],
        ["2", "Second", "2002-01-01", "Drama", "robot", "robot story", "20"],
        ["1", "First", "2001-01-01", "Action", "robot", "robot story", "30"],
        ["1", "Duplicate", "2001", "Action", "", "story", "100"],
        ["4", "No overview", "2004", "Action", "", "", "1000"],
        ["5", "No genres", "2005", "", "", "story", "1000"],
        ["bad", "Bad ID", "2006", "Action", "", "story", "1000"],
    ])
    output = tmp_path / "subset.csv.gz"
    assert import_catalog(archive, output=output, max_movies=2) == 2
    movies = pd.read_csv(output, keep_default_na=False)
    assert movies["movie_id"].tolist() == ["tmdb:1", "tmdb:2"]
    assert movies["year"].tolist() == [2001, 2002]
    assert movies["cast"].eq("").all()


def test_full_import_retains_rows_without_overviews(tmp_path):
    archive = archive_with_rows(tmp_path, [
        ["1", "Movie", "", "Action, Drama", "", "", ""],
    ])
    output = tmp_path / "full.csv.gz"
    assert import_catalog(archive, output=output) == 1
    assert pd.read_csv(output).iloc[0]["genres"] == "Action|Drama"


def test_failed_import_preserves_existing_catalog(tmp_path):
    archive = archive_with_rows(tmp_path, [["1", "Movie", "", "", "", "", "0"]])
    output = tmp_path / "subset.csv.gz"
    output.write_bytes(b"existing catalog")
    with pytest.raises(ValueError, match="no usable movies"):
        import_catalog(archive, output=output, max_movies=5)
    assert output.read_bytes() == b"existing catalog"
    assert not output.with_suffix(".tmp").exists()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_is_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="positive"):
        import_catalog(tmp_path / "unused.zip", max_movies=limit)


def test_full_catalog_supports_combined_filters(monkeypatch):
    monkeypatch.setattr(recommender, "DATA_FILE", recommender.KAGGLE_CATALOG)
    movies = recommender.load_movies()
    assert len(movies) > 1_000_000
    assert movies["movie_id"].is_unique
    results = recommender.recommend(
        movie_id="tmdb:603", genres=["Action"], min_year=2000, max_year=2020
    )
    assert len(results) == 5
    assert results["genres"].str.split("|").map(lambda genres: "Action" in genres).all()
    assert results["year"].astype(int).between(2000, 2020).all()
    assert "tmdb:603" not in results["movie_id"].tolist()
