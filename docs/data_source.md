# Movie catalog source

The 4,803-row `data/movies_tmdb.csv` is a reduced copy of Kaggle's [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), version 2 (last updated September 28, 2017; downloaded November 5, 2026). Kaggle says its source is [The Movie Database (TMDB)](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.

The source archive contains `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`. `scripts/build_tmdb_catalog.py` joins them by movie ID, discards rows with missing or repeated IDs or titles, and writes only the fields CineSense uses: ID, title, release year, genres, keywords, first five cast members, director, and plot overview. The original ZIP is not needed to run the app. To rebuild the CSV, download the dataset from Kaggle and run:

```bash
python scripts/build_tmdb_catalog.py path/to/tmdb-movie-metadata.zip
```

This is an older snapshot, so new films are absent and some metadata may be incomplete. Four records have no overview; the recommender can still compare their other fields. The original 30-movie sample remains available when `data/movies_tmdb.csv` is absent. Future API integration would need a TMDB API credential and its own caching, error handling, and attribution review.

See [TMDB's API FAQ](https://developer.themoviedb.org/docs/faq) for its current noncommercial use and attribution terms. The Kaggle dataset labels its license as "Other (specified in description)" and links to TMDB's terms; do not assume it is unrestricted public-domain data.
