"""Keep the core test suite independent of the locally downloaded catalog."""

import pandas as pd
import pytest

import poc_recommender


@pytest.fixture(scope="session")
def test_catalog(tmp_path_factory):
    """Read a small test sample from the single production catalog."""
    path = tmp_path_factory.mktemp("catalog") / "movies.csv"
    pd.read_csv(poc_recommender.KAGGLE_CATALOG, nrows=500).to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def use_bundled_catalog(monkeypatch, test_catalog):
    poc_recommender._catalog_and_vectors.cache_clear()
    monkeypatch.setattr(poc_recommender, "DATA_FILE", test_catalog)
    yield
    poc_recommender._catalog_and_vectors.cache_clear()
