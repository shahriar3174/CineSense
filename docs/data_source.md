# Movie catalog source

## Full Kaggle TMDB catalog

`data/movies_tmdb_kaggle.csv.gz` is a derived catalog of asaniczka's [Full TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies), sourced from TMDB. Downloaded November 13, 2026; Kaggle metadata reported version 1119, last updated November 13, 2026 at 06:05 UTC. The downloaded archive contains `TMDB_movie_dataset_v11.csv`.

Source ZIP SHA-256: `d20b15edb590c15b4b3b9345bfc64e718135a3779237b3a95b99d0f4fceb6cc7`.

Kaggle identifies the dataset license as [ODC Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/1-0/). Credit: asaniczka and The Movie Database (TMDB). This derived catalog retains that attribution; the app also displays source and license links. This product uses the TMDB API but is not endorsed or certified by TMDB.

The full catalog contains 1,537,332 records with valid unique numeric movie IDs and nonblank titles. The importer retains movie ID, title, year, genres, keywords, and overview. Cast and director are unavailable in this source and remain blank. Some records have missing genres or plot text, so recommendation coverage varies.

The compressed full catalog is stored in `data/movies_tmdb_kaggle.csv.gz` using Git LFS. Download it with `git lfs pull` after cloning. Rebuild it from the source archive using `python scripts/import_kaggle_catalog.py path/to/tmdb-movies-dataset.zip`.

This is a snapshot downloaded November 13, 2026. Refreshing requires importing a newer archive and committing the replacement. The app uses this full catalog exclusively.
