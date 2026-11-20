"""Prepared index compatibility and stale-index recovery."""

import json

import pytest
from scipy.sparse import save_npz

import poc_recommender as recommender


def test_prepared_index_preserves_ranking_and_skips_fitting(monkeypatch, tmp_path, test_catalog):
    monkeypatch.setattr(recommender, "KAGGLE_CATALOG", test_catalog)
    matrix_file = tmp_path / "vectors.npz"
    manifest_file = tmp_path / "vectors.json"
    monkeypatch.setattr(recommender, "VECTOR_FILE", matrix_file)
    monkeypatch.setattr(recommender, "VECTOR_MANIFEST", manifest_file)
    expected = recommender.recommend("Inception")
    movies, vectors = recommender._catalog_and_vectors()
    save_npz(matrix_file, vectors)
    manifest_file.write_text(json.dumps({
        "catalog_sha256": recommender.file_sha256(test_catalog),
        "vectors_sha256": recommender.file_sha256(matrix_file),
        "columns": vectors.shape[1],
    }))
    recommender._catalog_and_vectors.cache_clear()
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A prepared deployment must skip fitting.")
    monkeypatch.setattr(recommender, "_fit_vectors", unexpected_fit)
    actual = recommender.recommend("Inception")
    assert actual.equals(expected)


def test_stale_index_raises_actionable_catalog_error(monkeypatch, tmp_path, test_catalog):
    monkeypatch.setattr(recommender, "KAGGLE_CATALOG", test_catalog)
    manifest = tmp_path / "vectors.json"
    manifest.write_text(json.dumps({"catalog_sha256": "different-catalog"}))
    monkeypatch.setattr(recommender, "VECTOR_MANIFEST", manifest)
    with pytest.raises(recommender.CatalogError, match="out of date"):
        recommender.load_movies()
