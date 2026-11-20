"""Literal title matching for the movie autocomplete field."""


def find_movie_matches(query, movies, limit=30):
    """Return display labels and stable IDs for titles containing the typed text."""
    query = query.strip()
    if not query:
        return []
    matches = movies["title"].str.contains(query, case=False, regex=False)
    indices = movies.index[matches][:limit]
    return [
        (f"{row.title} ({row.year or 'year unknown'})", row.movie_id)
        for row in movies.loc[indices, ["movie_id", "title", "year"]].itertuples()
    ]
