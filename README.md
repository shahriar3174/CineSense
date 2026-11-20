# CineSense Proof of Concept

**Live demo:** [Open CineSense](https://cinesensemovierecommender.streamlit.app/) â€” deployed from this personal repository.

CineSense is a content-based movie recommender for the CIS 3296 course project. The application uses the full Kaggle TMDB catalog.

The program loads 1,537,332 movies from the Kaggle TMDB snapshot, converts genres, keywords, and plot overviews into TF-IDF vectors, and ranks similar movies with cosine similarity. The full compressed catalog is stored with Git LFS.

## Installation

You need:

- Python 3.10 or newer (developed with CPython 3.13.5)
- Windows 10/11, macOS, or Linux
- Internet access to clone the Git LFS catalog and install dependencies. Movie recommendations need no API key.

Check your Python version first:

| OS | Command |
| --- | --- |
| Windows | `py --version` |
| macOS / Linux | `python3 --version` |

If Python is missing, install it from [python.org](https://www.python.org/downloads/). On macOS you can also use Homebrew: `brew install python`.

## Build (one time)

CineSense is plain Python, so there is no compile step. "Building" means creating a virtual environment and installing the libraries.

Open a terminal **in the project folder** (the folder that contains `requirements.txt`).

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

When the environment is active, your prompt starts with `(.venv)`. The first install can take a few minutes.

## Download the catalog

Install [Git LFS](https://git-lfs.com/), then run these commands in the cloned repository:

```bash
git lfs install
git lfs pull
```

The application uses `data/movies_tmdb_kaggle.csv.gz`, approximately 182 MB compressed. This snapshot contains 1,537,332 movies. Cast and director are unavailable in this source and remain blank. Some records have incomplete metadata.

To refresh it manually, download the ZIP from [TMDB Movies on Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) and run:

```bash
python scripts/import_kaggle_catalog.py path/to/tmdb-movies-dataset.zip
```

After replacing the catalog, rebuild the prepared TF-IDF index locally before deploying:

```bash
python scripts/build_recommendation_index.py
```

Commit the updated catalog, `data/tmdb_vectors.npz`, `data/tmdb_catalog.feather.gz`, and `data/tmdb_vectors.json` together. The compressed catalog, prepared metadata, and sparse index are tracked with Git LFS. The app verifies their checksums before loading, so an old index cannot silently be used with a newer catalog. Updates require a new import, index build, and commit.

## Host on Streamlit Community Cloud

Connect this personal repository and deploy branch `main` with entry point `app.py`. Streamlit Community Cloud supports Git LFS. Confirm that the deployment downloads the full catalog and that its page reports **1,537,332 movies (Kaggle TMDB)**.

The deployed app maps prepared Arrow metadata and sparse TF-IDF arrays from disk to reduce startup memory. The full catalog and index have been verified locally. Validate memory usage and startup in the hosted deployment before the class demo. See [dataset provenance](docs/data_source.md).

## Run

Every time you open a new terminal, activate the environment again first (`source .venv/bin/activate` on macOS/Linux, `.\.venv\Scripts\Activate.ps1` on Windows).

**Command-line recommender**

```bash
python poc_recommender.py --movie-id tmdb:603
```

Select The Matrix by its stable ID, which distinguishes films sharing the same title:

```bash
python poc_recommender.py --movie-id tmdb:603
```

The title must match a movie in the active catalog exactly, ignoring capitalization (`"the matrix"` works, `"Matrix"` does not). Running the script with no title uses *The Matrix*. The web interface selects movies by stable movie ID, so films with the same title are distinct.

Optionally filter by any selected genre and an inclusive release-year range:

```bash
python poc_recommender.py --movie-id tmdb:603 --genre Action --genre "Science Fiction" --min-year 2000 --max-year 2020
```

Genre matching ignores capitalization. Repeat `--genre` to match any of the selected genres; omit it to include all genres. Filters apply before choosing the top five, using the same catalog similarity scores.

**Web interface**

```bash
python -m streamlit run app.py
```

Streamlit opens the page in your browser. It shows the active dataset and movie count. Pick a movie, optionally select genres and a release year range under **Filter recommendations**, then select **Get Recommendations**. Clear the genre selection to include all genres. Press **Ctrl + C** in the terminal to stop the server.

When you're done, run `deactivate` to leave the virtual environment.

## Where to watch

Choose a country under **Where to watch settings**, then request recommendations. CineSense displays subscription, free, ad-supported, rental, and purchase providers for your selected movie and its recommendations. The viewing-options button opens TMDB's watch page.

Availability comes from JustWatch through TMDB. Country listings are cached for one day and each movie's provider response for one hour. Check with the provider before paying, as listings can change.

Create an API credential in [TMDB account settings](https://www.themoviedb.org/settings/api). Save one of the following settings in `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your-api-key"
```

Alternatively, use `TMDB_ACCESS_TOKEN = "your-read-access-token"`. Environment variables with the same names are also supported. The secrets file is excluded from Git.

For Streamlit Community Cloud, add the credential through **App settings â†’ Secrets**. Recommendations continue to work when availability is unconfigured or TMDB cannot be reached. API requests use existing TMDB movie IDs.

## Tests

With pytest installed, run the test suite:

```bash
python -m pytest -q
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `python3: command not found` (macOS/Linux) or `py` not recognized (Windows) | Python is not installed or not on your PATH. Install it as described under Requirements. |
| `ModuleNotFoundError: No module named 'pandas'` (or `sklearn`, `streamlit`) | The virtual environment is not active. Activate it, then run the command again. If it still fails, rerun `python -m pip install -r requirements.txt`. |
| `streamlit: command not found` | Use `python -m streamlit run app.py` with the environment active. |
| PowerShell: "running scripts is disabled on this system" | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then activate again. |
| `ValueError: Movie not found: ...` | The title isn't in the active catalog. Search by title in the app or select a known movie ID. |
| `FileNotFoundError` for `requirements.txt` | Your terminal isn't in the project folder. `cd` into it and try again. |

## Files

- `poc_recommender.py` - core proof of concept
- `app.py` - minimal Streamlit interface
- `data/movies_tmdb_kaggle.csv.gz` - full movie catalog tracked by Git LFS
- `scripts/import_kaggle_catalog.py` - converts the Kaggle archive to the application catalog
- `docs/data_source.md` - source, attribution, and data limitations
- `requirements.txt` - required libraries

The included TMDB data is a historical snapshot, not a live API feed. See [data source notes](docs/data_source.md) before redistributing or expanding it.

Author: Mohammad Ahnaf Shahriar
