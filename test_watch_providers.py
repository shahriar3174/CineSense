"""Watch-provider authentication, caching, region handling, and safe failures."""

from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest

import watch_providers as providers


@pytest.fixture(autouse=True)
def clear_provider_cache():
    providers.get_movie_providers.clear()
    providers.get_regions.clear()
    yield
    providers.get_movie_providers.clear()
    providers.get_regions.clear()


def response(payload):
    return BytesIO(json.dumps(payload).encode())


def test_regions_use_english_names_and_sort():
    payload = {"results": [
        {"iso_3166_1": "US", "english_name": "United States"},
        {"iso_3166_1": "BD", "english_name": "Bangladesh"},
    ]}
    with patch.object(providers, "urlopen", return_value=response(payload)):
        assert list(providers.get_regions("example-key")) == ["BD", "US"]


def test_cache_reuses_all_countries_for_one_movie():
    payload = {"results": {"US": {"flatrate": [{"provider_name": "Example"}]}, "BD": {}}}
    with patch.object(providers, "urlopen", return_value=response(payload)) as request:
        first = providers.get_movie_providers("tmdb:603", "example-key")
        second = providers.get_movie_providers("tmdb:603", "example-key")
        assert first == second == payload["results"]
        assert request.call_count == 1
        assert request.call_args.args[0].full_url.endswith("movie/603/watch/providers?api_key=example-key")


def test_token_uses_authorization_header():
    with patch.object(providers, "urlopen", return_value=response({"results": {}})) as request:
        providers.get_movie_providers("tmdb:603", access_token="example-token")
        assert request.call_args.args[0].get_header("Authorization") == "Bearer example-token"
        assert "example-token" not in request.call_args.args[0].full_url


def test_payment_types_stay_separate_and_missing_country_is_empty():
    payload = {"US": {
        "flatrate": [{"provider_name": "Example"}, {"provider_name": "Example"}],
        "rent": [{"provider_name": "Example"}],
        "ads": [{"provider_name": "Ad Service"}],
        "link": "https://www.themoviedb.org/movie/603/watch?locale=US",
    }}
    groups, link = providers.country_offers(payload, "US")
    assert groups == {"Subscription": ["Example"], "Free with ads": ["Ad Service"], "Rent": ["Example"]}
    assert link.startswith("https://www.themoviedb.org/")
    assert providers.country_offers(payload, "BD") == ({}, "")


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_http_failures_hide_credentials_and_are_not_cached(status):
    error = HTTPError("https://example.test/?api_key=private", status, "failed", None, None)
    with patch.object(providers, "urlopen", side_effect=error) as request:
        for _ in range(2):
            with pytest.raises(providers.AvailabilityError) as caught:
                providers.get_movie_providers("tmdb:603", "private")
            assert "private" not in str(caught.value)
        assert request.call_count == 2


def test_connection_failure_is_actionable():
    with patch.object(providers, "urlopen", side_effect=URLError("network")):
        with pytest.raises(providers.AvailabilityError, match="Try again later"):
            providers.get_movie_providers("tmdb:603", "example-key")


@pytest.mark.parametrize("payload", [[], {"results": []}, {}])
def test_malformed_results_raise_controlled_error(payload):
    with patch.object(providers, "urlopen", return_value=response(payload)):
        with pytest.raises(providers.AvailabilityError):
            providers.get_movie_providers("tmdb:603", "example-key")


def test_missing_credentials_and_invalid_ids_make_no_requests():
    with patch.object(providers, "urlopen") as request:
        with pytest.raises(providers.AvailabilityError):
            providers.get_movie_providers("tmdb:603")
        with pytest.raises(providers.AvailabilityError):
            providers.get_movie_providers("../../secret", "example-key")
        request.assert_not_called()


@pytest.mark.parametrize("url", ["javascript:alert(1)", "https://unrelated.test/watch", "https://["])
def test_untrusted_watch_link_is_not_rendered(url):
    assert providers.country_offers({"US": {"link": url}}, "US") == ({}, "")
