import numpy as np
import pandas as pd
import time

from IPython.display import HTML
import pickle
import json
import recommenders as rc
import imdb
import streamlit as st
from st_aggrid import AgGrid

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

movies_pd = pd.read_csv('movies.csv', index_col=0)
movies = movies_pd['0'].to_list()

BEST_MOVIES = pd.read_csv("best_movies.csv")

BEST_MOVIES.rename(
    index=lambda x: x + 1,
    inplace=True
)
TITLES = ["---"] + list(BEST_MOVIES['title'].sort_values())

with open('distance_recommender_kp.pkl', 'rb') as file:
    model_nn = pickle.load(file)

with open('nmf_model_base_60.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# sidebar
with st.sidebar:
    # title
    st.title("It's movie time!")
    # image
    st.image('nahtloses/1154.jpg')
    # blank space
    st.write("")
    # selectbox
    page = st.selectbox(
        "What would you like to find?",
        [
            "--------",
            "Popular movies",
            "Rate some movies",
            "Recommended movies"
        ]
    )
    n = st.slider(
        label="How many movies would you like to see?",
        min_value=1,
        max_value=20
    )

if page == "--------":
    # slogan
    st.write("""
    *Movies are like magic tricks (Jeff Bridges)*
    """)
    # blank space
    st.write("")
    # image
    st.image('movie_pics.png')

##########################################################
# Popular Movies
##########################################################

elif page == "Popular movies":
    # title
    st.title("Popular Movies")
    col1, col2 = st.columns([10, 10])

    with col1:
        st.markdown("####")
        genre = st.checkbox("include genres")
    with col2:
        st.markdown("###")
        show_button = st.button(label="show movies")

    if genre:
        popular_movies = BEST_MOVIES[['title', 'genres']]
    else:
        popular_movies = BEST_MOVIES[['title']]

    st.markdown("###")
    if show_button:
        st.write(
            HTML(popular_movies.head(n).to_html(escape=False))
        )

##########################################################
# Rate Movies
##########################################################

elif page == "Rate some movies":
    # title
    st.title("Rate Movies")
    #
    col1, col2, col3 = st.columns([10, 1, 5])
    with col1:
        m1 = st.selectbox("movie 1", TITLES)
        st.write("")
        m2 = st.selectbox("movie 2", TITLES)
        st.write("")
        m3 = st.selectbox("movie 3", TITLES)
        st.write("")
        m4 = st.selectbox("movie 4", TITLES)
        st.write("")
        m5 = st.selectbox("movie 5", TITLES)

    with col3:
        r1 = st.slider(
            label="rating 1",
            min_value=1,
            max_value=5,
            value=3
        )
        r2 = st.slider(
            label="rating 2",
            min_value=1,
            max_value=5,
            value=3
        )
        r3 = st.slider(
            label="rating 3",
            min_value=1,
            max_value=5,
            value=3
        )
        r4 = st.slider(
            label="rating 4",
            min_value=1,
            max_value=5,
            value=3
        )
        r5 = st.slider(
            label="rating 5",
            min_value=1,
            max_value=5,
            value=3
        )

    query_movies = [m1, m2, m3, m4, m5]
    query_ratings = [r1, r2, r3, r4, r5]

    user_query = dict(zip(query_movies, query_ratings))

    # get user query
    st.markdown("###")
    user_query_button = st.button(label="save user query")
    if user_query_button:
        json.dump(
            user_query,
            open("user_query.json", 'w')
        )
        st.write("")
        st.write("user query saved successfully")

##########################################################
# Movie Recommendations
##########################################################
else:
    # title
    st.title("Movie Recommendations")
    col1, col2, col3 = st.columns([20, 1, 10])
    with col1:
        recommender = st.radio(
            "recommender type",
            ["NMF Recommender", "Distance Recommender"]
        )

    with col3:
        st.write("###")
        recommend_button = st.button(label="recommend movies")

    # load user query
    user_query = json.load(open("user_query.json"))

    if recommend_button:
        if recommender == "NMF Recommender":
            recommend = rc.recommend_nmf(user_query, loaded_model, n)
        elif recommender == "Distance Recommender":
            recommend = rc.recommend_neighborhood(user_query, model_nn, n)

        st.write(recommend)
        st.write(recommend['Title'].to_list())
        data = imdb.get_imdb(recommend['Title'].to_list())
        st.write(data)
        for movies in data:
            st.write(movies[0]['title'])
            st.write(movies[0]['description'])
            st.image(movies[0]['image'], width=200)
