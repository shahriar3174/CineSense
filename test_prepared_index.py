"""Prepared index compatibility and stale-index recovery."""

import json
import gzip
import shutil

import pyarrow.feather as feather
import pytest
from scipy.sparse import save_npz

import poc_recommender as recommender


def test_prepared_index_preserves_ranking_and_skips_fitting(monkeypatch, tmp_path, test_catalog):
    monkeypatch.setattr(recommender, "KAGGLE_CATALOG", test_catalog)
    matrix_file = tmp_path / "vectors.npz"
    manifest_file = tmp_path / "vectors.json"
    metadata_file = tmp_path / "metadata.feather.gz"
    monkeypatch.setattr(recommender, "METADATA_FILE", metadata_file)
    monkeypatch.setattr(recommender, "VECTOR_FILE", matrix_file)
    monkeypatch.setattr(recommender, "VECTOR_MANIFEST", manifest_file)
    expected = recommender.recommend("Inception")
    movies, vectors = recommender._catalog_and_vectors()
    save_npz(matrix_file, vectors)
    raw_metadata = tmp_path / "metadata.raw"
    feather.write_feather(movies.drop(columns="features"), raw_metadata, compression="uncompressed")
    with raw_metadata.open("rb") as source, gzip.open(metadata_file, "wb") as target:
        shutil.copyfileobj(source, target)
    manifest_file.write_text(json.dumps({
        "catalog_sha256": recommender.file_sha256(test_catalog),
        "vectors_sha256": recommender.file_sha256(matrix_file),
        "columns": vectors.shape[1],
        "metadata_sha256": recommender.file_sha256(metadata_file),
    }))
    recommender._catalog_and_vectors.cache_clear()
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A prepared deployment must skip fitting.")
    monkeypatch.setattr(recommender, "_fit_vectors", unexpected_fit)
    monkeypatch.setattr(recommender, "_read_catalog", unexpected_fit)
    actual = recommender.recommend("Inception")
    assert actual.to_dict("records") == expected.to_dict("records")


def test_stale_index_raises_actionable_catalog_error(monkeypatch, tmp_path, test_catalog):
    monkeypatch.setattr(recommender, "KAGGLE_CATALOG", test_catalog)
    manifest = tmp_path / "vectors.json"
    manifest.write_text(json.dumps({"catalog_sha256": "different-catalog"}))
    monkeypatch.setattr(recommender, "VECTOR_MANIFEST", manifest)
    with pytest.raises(recommender.CatalogError, match="out of date"):
        recommender.load_movies()
