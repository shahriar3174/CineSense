"""Prepare the full-catalog TF-IDF index before deploying to Streamlit."""

import json
import gzip
import shutil
from pathlib import Path
import sys

import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.sparse import save_npz
from poc_recommender import (
    KAGGLE_CATALOG, METADATA_FILE, VECTOR_FILE, VECTOR_MANIFEST,
    _read_catalog, _fit_vectors, file_sha256,
)


def main():
    catalog_sha = file_sha256(KAGGLE_CATALOG)
    movies = _read_catalog(KAGGLE_CATALOG)
    vectors = _fit_vectors(movies, large=True)
    if file_sha256(KAGGLE_CATALOG) != catalog_sha:
        raise ValueError("The catalog changed during indexing. Run the command again.")
    temporary = VECTOR_FILE.with_suffix(".tmp")
    try:
        with temporary.open("wb") as target:
            save_npz(target, vectors)
        manifest = {
            "catalog_sha256": catalog_sha,
            "vectors_sha256": file_sha256(temporary),
            "columns": vectors.shape[1],
            "rows": vectors.shape[0],
        }
        metadata_tmp = METADATA_FILE.with_suffix(".tmp")
        metadata_raw = METADATA_FILE.with_suffix(".raw")
        try:
            feather.write_feather(movies.drop(columns="features"), metadata_raw, compression="uncompressed")
            with metadata_raw.open("rb") as source, gzip.open(metadata_tmp, "wb", compresslevel=3) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            metadata_tmp.replace(METADATA_FILE)
        finally:
            metadata_tmp.unlink(missing_ok=True)
            metadata_raw.unlink(missing_ok=True)
        manifest["metadata_sha256"] = file_sha256(METADATA_FILE)
        temporary.replace(VECTOR_FILE)
        VECTOR_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {vectors.shape[0]:,} movies and {vectors.shape[1]:,} TF-IDF features.")


if __name__ == "__main__":
    main()
