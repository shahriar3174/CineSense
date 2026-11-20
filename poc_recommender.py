
"""Minimal CineSense proof of concept."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
from threading import Lock
import zipfile
from functools import lru_cache
from random import seed

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent / "data"
KAGGLE_CATALOG = DATA_DIR / "movies_tmdb_kaggle.csv.gz"
DATA_FILE = KAGGLE_CATALOG
VECTOR_FILE = DATA_DIR / "tmdb_vectors.npz"
VECTOR_MANIFEST = DATA_DIR / "tmdb_vectors.json"
_INDEX_LOCK = Lock()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def load_prepared_vectors(path: Path, checksum: str):
    """Map the prepared sparse arrays from disk instead of keeping copies in RAM."""
    directory = Path(tempfile.gettempdir()) / "cinesense-tfidf" / checksum
    with _INDEX_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            shape = tuple(np.load(archive.open("shape.npy"), allow_pickle=False))
            for name in ("data.npy", "indices.npy", "indptr.npy"):
                destination = directory / name
                if destination.exists() and destination.stat().st_size == archive.getinfo(name).file_size:
                    continue
                temporary = destination.with_suffix(".tmp")
                try:
                    with archive.open(name) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
    arrays = [np.load(directory / name, mmap_mode="r", allow_pickle=False)
              for name in ("data.npy", "indices.npy", "indptr.npy")]
    return csr_matrix(tuple(arrays), shape=shape, copy=False)


# ---------------------------------------------------------------------------
# Custom exception hierarchy â€” gives the UI layer enough context to show
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

    def __init__(
        self, *, has_year_filter: bool = False, has_genre_filter: bool = False
    ) -> None:
        self.has_year_filter = has_year_filter
        self.has_genre_filter = has_genre_filter
        suggestions = []
        if has_year_filter:
            suggestions.append("widening or removing the year filter")
        if has_genre_filter:
            suggestions.append("removing or changing the genre filter")
        hint = " Try " + (" or ".join(suggestions) or "choosing a different movie") + "."
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
    # Drop rows that have no title â€” they cannot be selected.
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


def _read_catalog(path: Path | None = None, *, build_features: bool = True) -> pd.DataFrame:
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
            "Run git lfs pull in the repository folder to download the full catalog."
        )

    try:
        if source.resolve() == KAGGLE_CATALOG.resolve():
            # Arrow strings and bounded CSV chunks avoid Python objects for millions of cells.
            chunks = pd.read_csv(
                source, dtype="string[pyarrow]", keep_default_na=False, chunksize=25000
            )
            movies = pd.concat(chunks, ignore_index=True)
        else:
            movies = pd.read_csv(source, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise EmptyCatalogError("The movie catalog contains no rows.") from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CatalogNotFoundError(
            "Cannot read the movie catalog. Check that it is a valid UTF-8 CSV."
        ) from exc

    movies = movies.fillna("")
    movies = _validate_catalog(movies, str(source))

    if not build_features:
        return movies

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


def _fit_vectors(movies: pd.DataFrame, *, large: bool = False):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        if large:
            vectorizer.set_params(dtype=np.float32, min_df=3)
        return vectorizer.fit_transform(movies["features"])
    except ValueError as exc:
        raise NoUsableFeaturesError("The catalog has no usable words to compare.") from exc


@lru_cache(maxsize=1)
def _catalog_and_vectors():
    """Load the active catalog and build TF-IDF once per process."""
    if DATA_FILE == KAGGLE_CATALOG and VECTOR_MANIFEST.exists():
        try:
            manifest = json.loads(VECTOR_MANIFEST.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Invalid index manifest.")
            if (
                manifest.get("catalog_sha256") != file_sha256(KAGGLE_CATALOG)
                or manifest.get("vectors_sha256") != file_sha256(VECTOR_FILE)
            ):
                raise ValueError("Index does not match the catalog.")
            movies = _read_catalog(build_features=False)
            pa.default_memory_pool().release_unused()
            vectors = load_prepared_vectors(
                VECTOR_FILE, manifest["vectors_sha256"]
            )
            if vectors.shape != (len(movies), manifest["columns"]):
                raise ValueError("Index dimensions do not match the catalog.")
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            raise CatalogError(
                "The prepared recommendation index is missing or out of date. "
                "Run python scripts/build_recommendation_index.py and restart the app."
            ) from error
        return movies, vectors

    movies = _read_catalog()
    return movies, _fit_vectors(movies, large=DATA_FILE == KAGGLE_CATALOG)


def load_movies(path: Path | None = None) -> pd.DataFrame:
    """Return a copy of the active catalog, or validate an explicit CSV."""
    if path is not None:
        movies = _read_catalog(path)
        _fit_vectors(movies, large=path.resolve() == KAGGLE_CATALOG.resolve())
        return movies
    return _catalog_and_vectors()[0].copy()


def available_genres() -> list[str]:
    """Return sorted, unique, nonblank genres from the active catalog."""
    movies, _ = _catalog_and_vectors()
    return sorted({
        genre.strip()
        for value in movies["genres"]
        for genre in value.split("|")
        if genre.strip()
    })


def shared_attributes(seed, other) -> dict:
    """Return the genres, keywords, cast, and director two movies have in common."""
    shared = {}
    for field in ["genres", "keywords", "cast"]:
        seed_values = set(seed[field].split("|")) - {""}
        other_values = set(other[field].split("|")) - {""}
        common = seed_values.intersection(other_values)
        if common:
            shared[field] = sorted(common)
    if seed["director"] and seed["director"] == other["director"]:
        shared["director"] = seed["director"]
    return shared

def recommend(
    title: str | None = None,
    limit: int = 5,
    min_year: int | None = None,
    max_year: int | None = None,
    movie_id: str | None = None,
    genres: list[str] | None = None,
) -> pd.DataFrame:
    """Rank similar movies with inclusive year and any-match genre filters."""
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
    if DATA_FILE == KAGGLE_CATALOG:
        # TF-IDF rows already have unit length, so their dot product is cosine similarity.
        scores = (vectors @ vectors[selected_index].T).toarray().ravel()
    else:
        scores = cosine_similarity(vectors[selected_index], vectors).ravel()
    eligible = scores > 0
    eligible[selected_index] = False

    has_year_filter = min_year is not None or max_year is not None
    if has_year_filter:
        years = pd.to_numeric(movies["year"], errors="coerce").to_numpy()
        if min_year is not None:
            eligible &= years >= min_year
        if max_year is not None:
            eligible &= years <= max_year

    has_genre_filter = bool(genres)
    if has_genre_filter:
        requested_genres = {genre.strip().casefold() for genre in genres}
        eligible &= movies["genres"].map(
            lambda value: any(
                genre.strip().casefold() in requested_genres
                for genre in value.split("|") if genre.strip()
            )
        ).to_numpy()

    eligible_indices = np.flatnonzero(eligible)
    if not eligible_indices.size:
        raise NoEligibleResultsError(
            has_year_filter=has_year_filter, has_genre_filter=has_genre_filter
        )

    order = np.argsort(-scores[eligible_indices], kind="stable")
    best_indices = eligible_indices[order[:limit]]

    results = movies.loc[
        best_indices, ["title", "year", "genres", "movie_id", "keywords", "cast", "director", "overview"]
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
        help="Movie title (defaults to The Matrix)"
    )

    parser.add_argument("--movie-id", help="Select a movie by its catalog ID")
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument(
        "--genre", action="append", dest="genres",
        help="Include movies matching any selected genre (repeatable)",
    )

    args = parser.parse_args()
    title = args.title or ("The Matrix" if args.movie_id is None else None)

    try:
        results = recommend(
            title=title,
            min_year=args.min_year,
            max_year=args.max_year,
            movie_id=args.movie_id,
            genres=args.genres,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\nCineSense recommendations for {args.movie_id or title}:\n")

    if results.empty:
        print("No matching recommendations found.")
        return

    if len(results) < 5:
        print(
            f"Only {len(results)} recommendation(s) found "
            f"for {args.movie_id or title}."
        )

    for number, movie in results.iterrows():
        print(
            f"{number + 1}. {movie['title']} ({movie['year']}) "
            f"- {movie['genres']} - score {movie['similarity']:.3f}"
        )


if __name__ == "__main__":
    main()
