import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import poc_recommender
from poc_recommender import _catalog_and_vectors, available_genres, recommend, NoEligibleResultsError

COLUMNS = ["title", "year", "genres", "keywords", "cast", "director", "overview"]

class RecommendTests(unittest.TestCase):
    def test_available_genres_are_sorted_unique_and_nonempty(self):
        genres = available_genres()
        self.assertTrue(genres)
        self.assertEqual(genres, sorted(set(genres)))
        self.assertTrue(all(genres))

    def test_action_filter(self):
        results = recommend("The Matrix", genres=["Action"])
        self.assertGreater(len(results), 0)
        self.assertTrue(all("Action" in value.split("|") for value in results["genres"]))

    def test_empty_genre_list_does_not_filter(self):
        self.assertTrue(recommend("The Matrix", genres=[]).equals(recommend("The Matrix")))

    def test_genre_filter_is_case_insensitive(self):
        self.assertTrue(recommend("The Matrix", genres=["aCtIoN"]).equals(
            recommend("The Matrix", genres=["Action"])
        ))

    def test_filter_preserves_scores_and_ranking_before_limit(self):
        all_results = recommend("The Matrix", limit=10000)
        expected = all_results[all_results["genres"].str.split("|").map(
            lambda genres: "Drama" in genres
        )].head(5).reset_index(drop=True)
        self.assertEqual(len(expected), 5)
        self.assertTrue(recommend("The Matrix", genres=["Drama"]).equals(expected))

    def test_returns_five_results_by_default(self):
        results = recommend("The Matrix")
        self.assertEqual(len(results), 5)

    def test_excludes_selected_movie(self):
        results = recommend("The Matrix")
        self.assertNotIn("The Matrix", results["title"].tolist())

    def test_results_are_sorted_highest_score_first(self):
        scores = recommend("The Matrix")["similarity"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_title_match_ignores_capitalization(self):
        self.assertTrue(recommend("the matrix").equals(recommend("The Matrix")))

    def test_unknown_title_raises_error(self):
        with self.assertRaises(ValueError):
            recommend("Not A Real Movie")

    def test_min_year_greater_than_max_year_raises_error(self):
        with self.assertRaises(ValueError):
            recommend("The Matrix", min_year=2020, max_year=2010)

    def test_year_filter_keeps_results_in_range(self):
        results = recommend("The Matrix", min_year=2014, max_year=2015)
        for year in results["year"]:
            self.assertTrue(2014 <= int(year) <= 2015)

    def test_repeated_requests_do_not_refit(self):
        _catalog_and_vectors.cache_clear()
        recommend("The Matrix")
        recommend("Inception")
        self.assertEqual(_catalog_and_vectors.cache_info().misses, 1)

    def test_cached_results_match_fresh_results(self):
        cached = recommend("Inception")
        _catalog_and_vectors.cache_clear()
        fresh = recommend("Inception")
        self.assertTrue(cached.equals(fresh))

class EdgeCaseTests(unittest.TestCase):

    def use_catalog(self, rows):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "catalog.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        patcher = patch.object(poc_recommender, "DATA_FILE", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        _catalog_and_vectors.cache_clear()
        self.addCleanup(_catalog_and_vectors.cache_clear)

    def test_tied_scores_keep_catalog_order(self):
        self.use_catalog([
            ["Seed",   "2000", "Space", "robot", "", "", "robot space"],
            ["Twin A", "2001", "Space", "robot", "", "", "robot space"],
            ["Twin B", "2002", "Space", "robot", "", "", "robot space"],
        ])
        results = recommend("Seed")
        self.assertEqual(results["title"].tolist(), ["Twin A", "Twin B"])

    def test_available_genres_strips_whitespace_and_empty_items(self):
        self.use_catalog([
            ["Seed", "2000", " Action |Drama|| ", "robot", "", "", ""],
            ["Other", "2001", "Drama|Comedy|Action", "robot", "", "", ""],
        ])
        self.assertEqual(available_genres(), ["Action", "Comedy", "Drama"])

    def test_genres_match_any_selected_genre_with_inclusive_year_bounds(self):
        self.use_catalog([
            ["Seed", "2000", "Action|Comedy", "robot", "", "", ""],
            ["Action only", "2001", " Action ", "robot", "", "", ""],
            ["Comedy only", "2002", "Comedy", "robot", "", "", ""],
            ["Both", "2003", "Action|Comedy", "robot", "", "", ""],
            ["Wrong genre", "2002", "Drama", "robot", "", "", ""],
            ["Unknown year", "", "Action", "robot", "", "", ""],
            ["Partial genre", "2002", "Actionable", "robot", "", "", ""],
        ])
        results = recommend("Seed", limit=10, genres=["Action", "Comedy"])
        self.assertEqual(set(results["title"]), {"Action only", "Comedy only", "Both", "Unknown year"})
        results = recommend("Seed", genres=["Action", "Comedy"], min_year=2001, max_year=2002)
        self.assertEqual(set(results["title"]), {"Action only", "Comedy only"})

    def test_genre_match_still_requires_positive_similarity(self):
        self.use_catalog([
            ["Seed", "2000", "Western", "cowboy", "", "", ""],
            ["Other", "2001", "Action", "robot", "", "", ""],
        ])
        with self.assertRaises(NoEligibleResultsError):
            recommend("Seed", genres=["Action"])

    def test_small_catalog_returns_fewer_than_five(self):
        self.use_catalog([
            ["Seed",  "2000", "Space", "robot", "", "", "robot"],
            ["Other", "2001", "Space", "alien", "", "", "alien"],
            ["Third", "2002", "Drama", "robot", "", "", "robot"],
        ])
        self.assertEqual(len(recommend("Seed")), 2)

    def test_no_positive_matches_raises_error(self):
        self.use_catalog([
            ["Seed",  "2000", "Western", "cowboy", "", "", "cowboy"],
            ["Other", "2001", "Space",   "alien",  "", "", "alien"],
        ])
        with self.assertRaises(NoEligibleResultsError):
            recommend("Seed")


class CliTests(unittest.TestCase):
    def test_cli_defaults_and_repeatable_genres(self):
        cases = [
            ([], "The Matrix", None, None),
            (["Inception"], "Inception", None, None),
            (["--movie-id", "tmdb:603"], None, "tmdb:603", None),
            (["--genre", "Action", "--genre", "Science Fiction"],
             "The Matrix", None, ["Action", "Science Fiction"]),
        ]
        for arguments, title, movie_id, genres in cases:
            with self.subTest(arguments=arguments), patch("sys.argv", ["poc_recommender.py", *arguments]), \
                    patch.object(poc_recommender, "recommend") as mocked, patch("builtins.print"):
                poc_recommender.main()
                mocked.assert_called_once_with(
                    title=title, movie_id=movie_id, min_year=None, max_year=None, genres=genres
                )


if __name__ == "__main__":
    unittest.main()
