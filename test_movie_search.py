import pandas as pd

from movie_search import find_movie_matches


def test_partial_titles_match_anywhere_without_case_sensitivity():
    movies = pd.DataFrame({
        "movie_id": ["tmdb:1", "tmdb:2", "tmdb:3"],
        "title": ["The Batman", "BATMAN Returns", "Arrival"],
        "year": ["2022", "1992", "2016"],
    })
    assert find_movie_matches("  batMAN ", movies) == [
        ("The Batman (2022)", "tmdb:1"),
        ("BATMAN Returns (1992)", "tmdb:2"),
    ]


def test_search_treats_regex_characters_literally():
    movies = pd.DataFrame({
        "movie_id": ["tmdb:1", "tmdb:2"],
        "title": ["Dr. No", "Batman"],
        "year": ["1962", "1989"],
    })
    assert find_movie_matches(".", movies) == [("Dr. No (1962)", "tmdb:1")]


def test_blank_and_unmatched_searches_have_no_suggestions():
    movies = pd.DataFrame({"movie_id": ["tmdb:1"], "title": ["Batman"], "year": ["1989"]})
    assert find_movie_matches(" ", movies) == []
    assert find_movie_matches("no such title", movies) == []


def test_duplicate_labels_keep_distinct_movie_ids_and_result_limit():
    movies = pd.DataFrame({
        "movie_id": ["tmdb:1", "tmdb:2", "tmdb:3"],
        "title": ["Batman", "Batman", "Batman"],
        "year": ["1989", "1989", ""],
    })
    assert [value for _, value in find_movie_matches("b", movies, limit=2)] == ["tmdb:1", "tmdb:2"]
    assert find_movie_matches("b", movies)[-1][0] == "Batman (year unknown)"
