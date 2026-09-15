"""Small Streamlit screen proving the planned web framework works."""

import streamlit as st

from poc_recommender import load_movies, recommend


st.set_page_config(page_title="CineSense", page_icon="🎬")
st.title("CineSense")
st.write("Choose a movie to see five content-based recommendations.")

movies = load_movies()
title = st.selectbox("Movie", movies["title"].tolist())

for _, movie in recommend(title).iterrows():
    st.write(
        f"**{movie['title']} ({movie['year']})**  "
        f"Similarity: {movie['similarity']:.3f}  "
        f"Genres: {movie['genres'].replace('|', ', ')}"
    )
