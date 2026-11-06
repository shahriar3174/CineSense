
"""Minimal CineSense proof of concept."""

from pathlib import Path
import argparse
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
LARGE_CATALOG = DATA_DIR / "movies_tmdb.csv"
SAMPLE_CATALOG = DATA_DIR / "movies_sample.csv"
DATA_FILE = LARGE_CATALOG if LARGE_CATALOG.exists() else SAMPLE_CATALOG

# ---------------------------------------------------------------------------
# Custom exception hierarchy — gives the UI layer enough context to show
# specific, actionable recovery messages (AC-1 and AC-2).
# ---------------------------------------------------------------------------

class CineSenseError(ValueError):
    """Base class for all CineSense application errors."""


class CatalogError(CineSenseError):
    """Problems loading or validating the movie catalog."""


class CatalogNotFoundError(CatalogError):
    """The catalog file is missing or unreadable."""


class InvalidSchemaError(CatalogError):
    """Required columns are absent from the catalog file."""


class EmptyCatalogError(CatalogError):
    """The catalog loaded successfully but contains zero usable rows."""


class NoUsableFeaturesError(CatalogError):
    """Every row's combined text-feature column is blank."""


class SelectionError(CineSenseError):
    """Problems with the user's movie selection."""


class MissingSelectionError(SelectionError):
    """No movie title was provided."""


class MovieNotFoundError(SelectionError):
    """The requested title does not match any catalog entry."""

    def __init__(self, title: str) -> None:
        self.title = title
        super().__init__(
            f"No movie matching \u201c{title}\u201d was found in the catalog. "
            "Check the spelling or choose a different title."
        )


class NoEligibleResultsError(CineSenseError):
    """Ranking succeeded but filters eliminated every candidate."""

    def __init__(self, *, has_year_filter: bool = False) -> None:
        self.has_year_filter = has_year_filter
        hint = (
            " Try widening or removing the year filter."
            if has_year_filter
            else " Try choosing a different movie."
        )
        super().__init__(
            "No recommendations match the current criteria." + hint
        )


# ---------------------------------------------------------------------------
# Required columns and helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"title", "year", "genres", "keywords", "cast",
                     "director", "overview"}


def _validate_catalog(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate schema, drop unusable rows, and check for content.

    Raises
    ------
    InvalidSchemaError
        If any required column is missing.
    EmptyCatalogError
        If no rows remain after dropping those without a title.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidSchemaError(
            f"The catalog at {source} is missing required columns: "
            f"{', '.join(sorted(missing))}. "
            "Expected columns: "
            f"{', '.join(sorted(_REQUIRED_COLUMNS))}."
        )
    # Drop rows that have no title — they cannot be selected.
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip().astype(bool)]
    if df.empty:
        raise EmptyCatalogError(
            f"The catalog at {source} contains no rows with a usable title."
        )
    df = df.reset_index(drop=True)
    if "movie_id" not in df.columns:
        df["movie_id"] = [f"sample:{i}" for i in range(len(df))]
    ids = df["movie_id"].str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise InvalidSchemaError("The catalog contains missing or duplicate movie IDs.")
    return df.assign(movie_id=ids)


def _read_catalog(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the movie catalog.

    Parameters
    ----------
    path : Path, optional
        Override the default data-file location. Useful for testing.

    Raises
    ------
    CatalogNotFoundError
        If the file does not exist or cannot be read.
    InvalidSchemaError
        If required columns are missing.
    EmptyCatalogError
        If the file has no usable rows.
    NoUsableFeaturesError
        If every combined text-feature string is empty.
    """
    source = path or DATA_FILE

    # --- AC-1: missing or unreadable data ---
    if not source.exists():
        raise CatalogNotFoundError(
            f"Cannot find the movie catalog at {source}. "
            "Make sure the data file is present and the path is correct."
        )

    try:
        movies = pd.read_csv(source, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise EmptyCatalogError("The movie catalog contains no rows.") from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CatalogNotFoundError(
            "Cannot read the movie catalog. Check that it is a valid UTF-8 CSV."
        ) from exc

    movies = movies.fillna("")
    movies = _validate_catalog(movies, str(source))

    # Build the combined text-feature column.
    movies["features"] = (
        movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["keywords"].str.replace("|", " ", regex=False)
        + " "
        + movies["cast"].str.replace("|", " ", regex=False)
        + " "
        + movies["director"]
        + " "
        + movies["overview"]
    )

    # --- AC-1: data with no usable text features ---
    if movies["features"].str.strip().eq("").all():
        raise NoUsableFeaturesError(
            "The catalog loaded but every movie's combined metadata is empty, "
            "so recommendations cannot be computed. Check that the data file "
            "contains genres, keywords, cast, director, or overview values."
        )

    return movies


def _fit_vectors(movies: pd.DataFrame):
    try:
        return TfidfVectorizer(stop_words="english").fit_transform(movies["features"])
    except ValueError as exc:
        raise NoUsableFeaturesError("The catalog has no usable words to compare.") from exc


@lru_cache(maxsize=1)
def _catalog_and_vectors():
    """Load the active catalog and build TF-IDF once per process."""
    movies = _read_catalog()
    return movies, _fit_vectors(movies)


def load_movies(path: Path | None = None) -> pd.DataFrame:
    """Return a copy of the active catalog, or validate an explicit CSV."""
    if path is not None:
        movies = _read_catalog(path)
        _fit_vectors(movies)
        return movies
    return _catalog_and_vectors()[0].copy()


def recommend(
    title: str | None = None,
    limit: int = 5,
    min_year: int | None = None,
    max_year: int | None = None,
    movie_id: str | None = None,
) -> pd.DataFrame:
    """Rank similar movies by stable ID or title, with optional year filtering."""
    selection = movie_id if movie_id is not None else title
    if not selection or not selection.strip():
        raise MissingSelectionError(
            "No movie was selected. Please choose a movie from the catalog."
        )

    if min_year is not None and max_year is not None and min_year > max_year:
        raise ValueError("Minimum year cannot exceed maximum year.")

    movies, vectors = _catalog_and_vectors()
    if movie_id is not None:
        matches = movies.index[movies["movie_id"] == movie_id]
    else:
        matches = movies.index[movies["title"].str.casefold() == selection.casefold()]
    if matches.empty:
        raise MovieNotFoundError(selection)
    if len(matches) > 1:
        raise ValueError(f"More than one movie is titled {title!r}; select its movie ID.")

    selected_index = matches[0]
    scores = cosine_similarity(vectors[selected_index], vectors).flatten()

    # Filter eligible movies by release year.
    years = pd.to_numeric(movies["year"], errors="coerce")

    has_year_filter = min_year is not None or max_year is not None
    eligible_indices = []

    for index in range(len(movies)):
        if index == selected_index:
            continue

        if scores[index] <= 0:
            continue

        year = years.iloc[index]

        if has_year_filter:
            if pd.isna(year):
                continue

        if min_year is not None and year < min_year:
            continue

        if max_year is not None and year > max_year:
            continue

        eligible_indices.append(index)

    # --- AC-2: no eligible results ---
    if not eligible_indices:
        raise NoEligibleResultsError(has_year_filter=has_year_filter)

    # Rank only eligible movies, then select the top results.
    best_indices = sorted(
        eligible_indices,
        key=lambda index: (-scores[index], index)
    )[:limit]

    results = movies.loc[
        best_indices, ["title", "year", "genres"]
    ].copy()

    results["similarity"] = scores[best_indices].round(3)

    return results.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CineSense movie recommendations"
    )

    parser.add_argument(
        "title",
        nargs="?",
        default="The Matrix"
    )

    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)

    args = parser.parse_args()

    try:
        results = recommend(
            args.title,
            min_year=args.min_year,
            max_year=args.max_year
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\nCineSense recommendations for {args.title}:\n")

    if results.empty:
        print("No matching recommendations found.")
        return

    for number, movie in results.iterrows():
        print(
            f"{number + 1}. {movie['title']} ({movie['year']}) "
            f"- {movie['genres']} - score {movie['similarity']:.3f}"
        )


if __name__ == "__main__":
    main()
