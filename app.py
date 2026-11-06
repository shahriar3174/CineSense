"""Accessible Streamlit interface for CineSense with clear error recovery.

Addresses the "Provide an accessible interface with clear error recovery"
feature from the product backlog, covering all five acceptance criteria:

AC-1  Readable recovery messages for catalog-level failures.
AC-2  Distinct messages for missing selections, unknown movies, and
      no eligible results, with actionable next steps.
AC-3  Stale results are never shown after a failed new request.
AC-4  All controls are keyboard-accessible via native Streamlit widgets;
      detail expansion uses st.expander (no keyboard traps).
AC-5  Meaningful labels, headings, text-based scores, and a layout that
      remains readable at 200 % zoom.
"""

import streamlit as st

from poc_recommender import (
    DATA_FILE,
    LARGE_CATALOG,
    CatalogError,
    CineSenseError,
    MissingSelectionError,
    MovieNotFoundError,
    NoEligibleResultsError,
    load_movies,
    recommend,
)

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(page_title="CineSense", page_icon="🎬")

# ── AC-5: Meaningful heading ─────────────────────────────────────────
st.title("🎬 CineSense")
st.markdown(
    "Choose a movie you like, then press **Get Recommendations** "
    "to see five similar films ranked by content similarity."
)

# ── Load catalog with AC-1 error recovery ────────────────────────────
try:
    movies = load_movies()
except CatalogError:
    # AC-1: readable recovery message; no traceback shown to user.
    st.error(
        "**Unable to load the movie catalog.**\n\n"
        "Please make sure the data file is present and correctly formatted, "
        "then reload this page."
    )
    st.stop()

# ── AC-5: labeled input / AC-4: keyboard-accessible selectbox ────────
st.subheader("Select a Movie", anchor="select-movie")

using_tmdb = DATA_FILE == LARGE_CATALOG
catalog_name = "TMDB 5000" if using_tmdb else "sample catalog"
st.caption(f"Catalog: {len(movies):,} movies ({catalog_name}).")
if using_tmdb:
    st.caption("Data from [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.")

labels = {
    row.movie_id: f"{row.title} ({row.year or 'year unknown'})"
    for row in movies.itertuples()
}
movie_id = st.selectbox(
    "Movie title",
    options=list(labels),
    format_func=labels.get,
    index=None,
    placeholder="Start typing or browse to select a movie…",
    help="Pick the movie you want recommendations for.",
)

# ── Optional year filter (keyboard-accessible columns) ────────────────
with st.expander("🔎 Filter by release year (optional)", expanded=False):
    col_min, col_max = st.columns(2)
    min_year = col_min.number_input(
        "Earliest release year",
        min_value=1888,
        max_value=2100,
        value=None,
        step=1,
        placeholder="e.g. 1990",
        help="Include only movies released in or after this year.",
    )
    max_year = col_max.number_input(
        "Latest release year",
        min_value=1888,
        max_value=2100,
        value=None,
        step=1,
        placeholder="e.g. 2020",
        help="Include only movies released in or before this year.",
    )

# ── Recommendation trigger ────────────────────────────────────────────
get_recs = st.button("Get Recommendations", type="primary")

# ── AC-3: clear any previous results before attempting a new request ──
# We use a flag so that stale results from a prior successful run are
# never displayed alongside or after a new error message.
if get_recs:
    # AC-2: missing selection
    if movie_id is None:
        st.warning(
            "**No movie selected.** "
            "Please choose a movie from the dropdown above, "
            "then press **Get Recommendations** again."
        )
        st.stop()

    # ── Run the recommender ───────────────────────────────────────────
    st.divider()
    st.subheader("Recommendations", anchor="recommendations")

    try:
        results = recommend(
            movie_id=movie_id,
            min_year=int(min_year) if min_year is not None else None,
            max_year=int(max_year) if max_year is not None else None,
        )
    except MovieNotFoundError as exc:
        # AC-2: unknown movie ID — should be very rare with a selectbox,
        # but can happen if the catalog changes between page loads.
        st.error(
            f"**Movie not found:** {exc.title}\n\n"
            "The catalog may have changed. "
            "Please reload the page and choose another movie."
        )
        st.stop()
    except NoEligibleResultsError as exc:
        # AC-2: no eligible results — actionable hint
        st.info(
            f"**No results found.** {exc}\n\n"
            "You can widen or remove year filters using the "
            "**Filter by release year** section above, "
            "or try a different movie."
        )
        st.stop()
    except CineSenseError as exc:
        # AC-1: catch-all for any other CineSense error
        st.error(
            "**Something went wrong while generating recommendations.**\n\n"
            f"{exc}\n\n"
            "Please try again or choose a different movie."
        )
        st.stop()
    except ValueError as exc:
        st.error(
            f"**Invalid filter settings.** {exc}\n\n"
            "Please correct the year range and try again."
        )
        st.stop()

    # ── Display results ───────────────────────────────────────────────
    # AC-5: text-based scores and headings; readable at 200 % zoom.
    result_count = len(results)
    st.markdown(
        f"Showing **{result_count}** recommendation{'s' if result_count != 1 else ''} "
        f"based on **{labels[movie_id]}**:"
    )

    for rank, (_, movie) in enumerate(results.iterrows(), start=1):
        genres_display = movie["genres"].replace("|", ", ")
        score_pct = f"{movie['similarity'] * 100:.1f}"

        # AC-4: st.expander is keyboard-accessible and has no traps.
        # AC-5: heading uses rank and title; score communicated as text.
        with st.expander(
            f"#{rank} — {movie['title']} ({movie['year']})  ·  "
            f"Similarity: {score_pct} %",
            expanded=(rank <= 3),
        ):
            st.markdown(
                f"**Genres:** {genres_display}  \n"
                f"**Similarity score:** {movie['similarity']:.3f} "
                f"({score_pct} %)"
            )
            st.caption(
                "Similarity is based on shared genres, keywords, cast, "
                "director, and plot text — it is not a quality rating or "
                "guarantee of enjoyment."
            )

    # ── Fewer-than-five note ──────────────────────────────────────────
    if result_count < 5:
        st.info(
            f"Only {result_count} recommendation{'s' if result_count != 1 else ''} "
            "met the current criteria. "
            "Try removing year filters or choosing a different movie "
            "to see more results."
        )

# ── Method & dataset information (always visible) ─────────────────────
st.divider()
with st.expander("ℹ️ About CineSense", expanded=False):
    st.markdown(
        "**How it works:**  \n"
        "CineSense combines each movie's genres, keywords, cast, director, "
        "and plot overview into a text profile, converts profiles to "
        "[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectors, "
        "and ranks other movies by "
        "[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). "
        "Higher scores mean more overlapping text features — not a "
        "prediction of personal enjoyment.\n\n"
        "**Dataset:**  \n"
        f"Currently using the {catalog_name} with "
        f"**{len(movies)}** movies included in the repository. "
        "No API key, external download, or personal viewing history "
        "is used.\n\n"
        "**Keyboard navigation:**  \n"
        "All controls — the movie selector, year filters, the "
        "recommendation button, and detail expanders — are reachable "
        "with **Tab** / **Shift + Tab** and operable with **Enter** or "
        "**Space**."
    )
