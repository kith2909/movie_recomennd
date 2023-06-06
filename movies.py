# import std libraries
import numpy as np
import pandas as pd
import time

from IPython.display import HTML
import pickle
import json
import recommenders as rc

import streamlit as st
from st_aggrid import AgGrid

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

movies_pd = pd.read_csv('movies.csv', index_col=0)
movies = movies_pd['0'].to_list()


def recommend_neighborhood(new_user_query, model, ratings, k=10, mean_input=True):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model.
    Returns a list of k movie ids.
    """
    new_user_dataframe = pd.DataFrame(new_user_query, columns=movies, index=['new_user'])
    # construct a user vector

    if mean_input:
        mn = ratings.mean(axis=1)
        mn = mn.mean()
    else:
        mn = 0

    new_user_dataframe_imputed = new_user_dataframe.fillna(mn)

    similarity_scores, neighbor_ids = model.kneighbors(
        new_user_dataframe_imputed,
        n_neighbors=5,
        return_distance=True
    )

    # sklearn returns a list of predictions
    # extract the first and only value of the list

    neighbors_df = pd.DataFrame(
        data={'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )

    # find n neighbors
    neighborhood = ratings.iloc[neighbor_ids[0]]
    neighborhood_filter = neighborhood#.drop(new_user_query.keys(), axis=1)
    # calculate their average rating

    df_score = neighborhood_filter.sum()
    df_score_ranked = df_score.sort_values(ascending=False).index

    # 3. ranking
    recommendations = df_score_ranked[:5]
    # filter out movies allready seen by the user

    # return the top-k highst rated movie ids or titles

    return recommendations


def recommend_nmf(new_user_query, model, ratings, k=10, mean_imput=True):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model.
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    new_user_dataframe = pd.DataFrame(new_user_query, columns=movies, index=['new_user'])
    # construct a user vector
    if mean_imput:
        mn = ratings.mean(axis=1)
        mn = mn.mean()
    else:
        mn = 0

    new_user_dataframe_imputed = new_user_dataframe.fillna(mn)

    P_new_user_matrix = model.transform(new_user_dataframe_imputed)

    P_new_user = pd.DataFrame(P_new_user_matrix,
                              columns=model.get_feature_names_out(),
                              index=['new_user'])

    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix, columns=movies, index=model.get_feature_names_out())

    R_hat_new_user_matrix = np.dot(P_new_user, Q)

    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                                  columns=movies,
                                  index=['new_user'])
    new_user_query.keys()

    R_hat_new_user_filter = R_hat_new_user.drop(new_user_query.keys(), axis=1)

    R_hat_new_user.T.sort_values(by=['new_user'], ascending=False).index.tolist()
    # return the top-k highst rated movie ids or titles
    ranked_f = R_hat_new_user_filter.T.sort_values(by=['new_user'], ascending=False)

    recommendation = ranked_f[:k]

    return recommendation



BEST_MOVIES = pd.read_csv("best_movies.csv")
BEST_MOVIES.rename(
    index=lambda x: x + 1,
    inplace=True
)
TITLES = ["---"] + list(BEST_MOVIES['title'].sort_values())

with open('distance_recommender_kp.pkl', 'rb') as file:
    DISTANCE_MODEL = pickle.load(file)

with open('nmf_model_base.pkl', 'rb') as file:
    NMF_MODEL = pickle.load(file)

# sidebar
with st.sidebar:
    # title
    st.title("It's movie time!")
    # image
    st.image('movie_time.jpg')
    # blank space
    st.write("")
    # selectbox
    page = st.selectbox(
        "what would you like?",
        [
            "--------",
            "Popular movies",
            "Rate some movies",
            "Recommended movies"
        ]
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

elif page == "popular movies":
    # title
    st.title("Popular Movies")
    col1, col2, col3, col4 = st.columns([10, 1, 5, 5])
    with col1:
        n = st.slider(
            label="how many movies?",
            min_value=1,
            max_value=10
        )
    with col3:
        st.markdown("####")
        genre = st.checkbox("include genres")
    with col4:
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

elif page == "rate some movies":
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
    col1, col2, col3, col4, col5 = st.columns([1, 5, 1, 5, 1])
    with col2:
        recommender = st.radio(
            "recommender type",
            ["NMF Recommender", "Distance Recommender"]
        )
    with col4:
        st.write("###")
        recommend_button = st.button(label="recommed movies")

    # load user query
    user_query = json.load(open("user_query.json"))

    #rc.recommend_neighborhood(user_query, DISTANCE_MODEL, BEST_MOVIES)
    #rc.recommend_nmf(user_query, NMF_MODEL, BEST_MOVIES)



