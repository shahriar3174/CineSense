"""Tests for CineSense error recovery and accessibility helpers.

Covers the acceptance criteria of the 'Provide an accessible interface
with clear error recovery' feature, verifying that the recommender raises
the correct exception type (with an actionable message) for each failure
mode, rather than producing an unhandled traceback.
"""

import textwrap
from pathlib import Path

import pytest

from poc_recommender import (
    CatalogNotFoundError,
    EmptyCatalogError,
    InvalidSchemaError,
    MissingSelectionError,
    MovieNotFoundError,
    NoEligibleResultsError,
    NoUsableFeaturesError,
    load_movies,
    recommend,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, content: str) -> Path:
    """Write *content* to a temporary CSV and return its path."""
    p = tmp_path / "movies.csv"
    p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return p


# ── AC-1: catalog-level recovery messages ─────────────────────────────

class TestCatalogRecovery:
    """AC-1: readable recovery for missing data, bad schema, empty
    catalog, and data with no usable text features."""

    def test_missing_file_raises_catalog_not_found(self, tmp_path):
        with pytest.raises(CatalogNotFoundError, match="Cannot find"):
            load_movies(tmp_path / "nonexistent.csv")

    def test_unreadable_file_raises_catalog_not_found(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x80\x81\x82")  # not valid CSV
        with pytest.raises(CatalogNotFoundError, match="Cannot read"):
            load_movies(bad)

    def test_missing_columns_raises_invalid_schema(self, tmp_path):
        p = _write_csv(tmp_path, "title,year\nFoo,2020")
        with pytest.raises(InvalidSchemaError, match="missing required columns"):
            load_movies(p)

    def test_empty_catalog_raises_empty_catalog_error(self, tmp_path):
        p = _write_csv(tmp_path, """\
            title,year,genres,keywords,cast,director,overview
            ,,,,,,
        """)
        with pytest.raises(EmptyCatalogError, match="no rows"):
            load_movies(p)

    def test_no_features_raises_no_usable_features(self, tmp_path):
        p = _write_csv(tmp_path, """\
            title,year,genres,keywords,cast,director,overview
            Blank Movie,2020,,,,,
        """)
        with pytest.raises(NoUsableFeaturesError, match="empty"):
            load_movies(p)


# ── AC-2: selection and result-level messages ─────────────────────────

class TestSelectionRecovery:
    """AC-2: distinguish missing selections, unknown movie IDs,
    and no eligible results."""

    def test_none_title_raises_missing_selection(self):
        with pytest.raises(MissingSelectionError, match="No movie was selected"):
            recommend(None)

    def test_empty_title_raises_missing_selection(self):
        with pytest.raises(MissingSelectionError, match="No movie was selected"):
            recommend("")

    def test_whitespace_title_raises_missing_selection(self):
        with pytest.raises(MissingSelectionError, match="No movie was selected"):
            recommend("   ")

    def test_unknown_title_raises_movie_not_found(self):
        with pytest.raises(MovieNotFoundError, match="No movie matching"):
            recommend("Definitely Not A Real Movie Title 12345")

    def test_movie_not_found_includes_title(self):
        try:
            recommend("Ghost Title")
        except MovieNotFoundError as exc:
            assert exc.title == "Ghost Title"

    def test_no_eligible_results_with_year_filter(self):
        """Year range that eliminates every candidate."""
        with pytest.raises(NoEligibleResultsError, match="year filter"):
            recommend("The Matrix", min_year=2099, max_year=2100)

    def test_no_eligible_results_hint_differs_without_filter(self):
        """Without a year filter the hint should suggest a different movie."""
        # All movies have *some* similarity to each other in the sample,
        # so this is hard to trigger naturally. We test the exception
        # constructor directly.
        exc = NoEligibleResultsError(has_year_filter=False)
        assert "different movie" in str(exc)
        assert "year filter" not in str(exc)

    @pytest.mark.parametrize("year_bounds", [{}, {"min_year": 2000, "max_year": 2020}])
    def test_unmatched_genre_has_actionable_active_filter_guidance(self, year_bounds):
        with pytest.raises(NoEligibleResultsError) as caught:
            recommend("The Matrix", genres=["Not a real genre"], **year_bounds)
        exc = caught.value
        assert exc.has_genre_filter
        assert exc.has_year_filter == bool(year_bounds)
        assert "removing or changing the genre filter" in str(exc)
        assert ("year filter" in str(exc)) == bool(year_bounds)


# ── Normal operation still works ──────────────────────────────────────

class TestHappyPath:
    """Ensure the refactored code still produces correct results."""

    def test_default_recommendation(self):
        results = recommend("The Matrix")
        assert len(results) == 5
        assert "title" in results.columns
        assert "similarity" in results.columns

    def test_case_insensitive_lookup(self):
        results = recommend("the matrix")
        assert len(results) == 5

    def test_year_filter(self):
        results = recommend("The Matrix", min_year=2010, max_year=2020)
        assert len(results) > 0
        for _, row in results.iterrows():
            assert 2010 <= int(row["year"]) <= 2020

    def test_invalid_year_range(self):
        with pytest.raises(ValueError, match="Minimum year cannot exceed"):
            recommend("The Matrix", min_year=2020, max_year=2010)
