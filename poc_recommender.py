"""Minimal CineSense proof of concept."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path(__file__).parent / "data" / "movies_sample.csv"


def load_movies() -> pd.DataFrame:
    """Load the included demonstration catalog."""
    movies = pd.read_csv(DATA_FILE).fillna("")
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
    return movies


def recommend(title: str, limit: int = 5) -> pd.DataFrame:
    """Return movies with metadata most similar to the selected title."""
    movies = load_movies()
    title_matches = movies.index[movies["title"].str.casefold() == title.casefold()]
    if title_matches.empty:
        raise ValueError(f"Movie not found: {title}")

    vectors = TfidfVectorizer(stop_words="english").fit_transform(movies["features"])
    scores = cosine_similarity(vectors[title_matches[0]], vectors).flatten()
    best_indices = scores.argsort()[::-1]
    best_indices = [index for index in best_indices if index != title_matches[0]][:limit]

    results = movies.loc[best_indices, ["title", "year", "genres"]].copy()
    results["similarity"] = scores[best_indices].round(3)
    return results.reset_index(drop=True)


def main() -> None:
    title = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The Matrix"
    print(f"\nCineSense recommendations for {title}:\n")
    for number, movie in recommend(title).iterrows():
        print(
            f"{number + 1}. {movie['title']} ({movie['year']}) "
            f"- {movie['genres']} - score {movie['similarity']:.3f}"
        )


if __name__ == "__main__":
    main()
