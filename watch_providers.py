"""Country-specific movie availability from TMDB, powered by JustWatch."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import streamlit as st


class AvailabilityError(ValueError):
    """A watch-provider request could not be completed."""


def _request(path: str, api_key: str, access_token: str) -> dict:
    if not api_key and not access_token:
        raise AvailabilityError("Where to watch has not been configured.")
    headers = {"Accept": "application/json"}
    url = "https://api.themoviedb.org/3/" + path
    if access_token:
        headers["Authorization"] = "Bearer " + access_token
    else:
        url += "?" + urlencode({"api_key": api_key})
    try:
        with urlopen(Request(url, headers=headers), timeout=6) as response:
            data = json.load(response)
    except HTTPError as exc:
        messages = {
            401: "The streaming availability credential needs to be updated.",
            403: "TMDB denied access to streaming availability.",
            404: "TMDB has no watch-provider record for this movie.",
            429: "Streaming availability has reached its request limit. Try again later.",
        }
        raise AvailabilityError(
            messages.get(exc.code, "Streaming availability is temporarily unavailable.")
        ) from None
    except (URLError, OSError, TimeoutError):
        raise AvailabilityError(
            "Could not connect to streaming availability. Try again later."
        ) from None
    except (ValueError, UnicodeError):
        raise AvailabilityError("TMDB returned an unreadable availability response.") from None
    if not isinstance(data, dict):
        raise AvailabilityError("TMDB returned an unexpected availability response.")
    return data


@st.cache_data(ttl=86400, max_entries=4, show_spinner=False)
def get_regions(api_key: str = "", access_token: str = "") -> dict[str, str]:
    """Return supported country codes and English names, cached for one day."""
    results = _request("watch/providers/regions", api_key, access_token).get("results")
    if not isinstance(results, list):
        raise AvailabilityError("TMDB returned an unexpected country list.")
    regions = {
        row["iso_3166_1"]: row["english_name"]
        for row in results
        if isinstance(row, dict)
        and isinstance(row.get("iso_3166_1"), str)
        and re.fullmatch("[A-Z]{2}", row["iso_3166_1"])
        and isinstance(row.get("english_name"), str)
        and row["english_name"].strip()
    }
    if not regions:
        raise AvailabilityError("TMDB returned an empty country list.")
    return dict(sorted(regions.items(), key=lambda item: item[1]))


@st.cache_data(ttl=3600, max_entries=1024, show_spinner=False)
def get_movie_providers(
    movie_id: str, api_key: str = "", access_token: str = ""
) -> dict:
    """Fetch all country offers once per movie, cached for one hour."""
    if not isinstance(movie_id, str) or not re.fullmatch(r"tmdb:[1-9]\d*", movie_id):
        raise AvailabilityError("This movie has no valid TMDB ID for availability lookup.")
    results = _request(
        f"movie/{movie_id.split(':')[1]}/watch/providers", api_key, access_token
    ).get("results")
    if not isinstance(results, dict):
        raise AvailabilityError("TMDB returned unexpected watch-provider data.")
    return results


def country_offers(results: dict, country: str) -> tuple[dict[str, list[str]], str]:
    """Group a country's providers by payment type and validate the watch link."""
    region = results.get(country, {})
    if not isinstance(region, dict):
        raise AvailabilityError("TMDB returned unexpected country availability.")
    groups = {}
    for category, label in (
        ("flatrate", "Subscription"), ("free", "Free"),
        ("ads", "Free with ads"), ("rent", "Rent"), ("buy", "Buy"),
    ):
        rows = region.get(category, [])
        if not isinstance(rows, list):
            raise AvailabilityError("TMDB returned unexpected provider listings.")
        names = sorted({
            row["provider_name"].strip()
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("provider_name"), str)
            and row["provider_name"].strip()
        })
        if names:
            groups[label] = names
    link = region.get("link", "")
    if not isinstance(link, str):
        link = ""
    try:
        parsed = urlsplit(link)
    except ValueError:
        return groups, ""
    if parsed.scheme != "https" or parsed.netloc not in ("www.themoviedb.org", "themoviedb.org"):
        link = ""
    return groups, link
