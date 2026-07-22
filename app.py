
import streamlit as st
from model import recommend, adf

st.title("Anime Recommendation System")

selected_anime = st.selectbox(
    "Choose an Anime",
    sorted(adf['name'].unique())
)

if st.button("Recommend"):
    recommendations = recommend(selected_anime)

    st.subheader("Recommended Anime")

    for anime in recommendations:
        st.write(anime)