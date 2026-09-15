# CineSense Proof of Concept

CineSense is a small content-based movie recommender for the CIS 3296 initial project proposal. This repository only proves that the planned tools work together; it is not the finished semester project.

The program loads the included sample CSV with pandas, converts movie metadata into TF-IDF vectors with scikit-learn, and ranks similar movies with cosine similarity. A minimal Streamlit page demonstrates the planned interface.

## Environment

- 64-bit Windows 10 or Windows 11
- CPython 3.10 or newer
- Developed and tested with CPython 3.13.5

## Run

Open PowerShell in this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python poc_recommender.py "The Matrix"
```

To open the small web interface:

```powershell
streamlit run app.py
```

The included CSV is a hand-written demonstration sample, so the POC runs without an API key or external download. The proposed full project will use a larger TMDB-derived dataset.

## Files

- `poc_recommender.py` - core proof of concept
- `app.py` - minimal Streamlit interface
- `data/movies_sample.csv` - small demonstration catalog
- `requirements.txt` - required libraries

Author: Mohammad Ahnaf Shahriar
