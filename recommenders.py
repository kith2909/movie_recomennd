import numpy as np
from sklearn.decomposition import NMF
import pandas as pd

movies_pd = pd.read_csv("movies.csv", index_col=0)
movies = movies_pd['0'].to_list()

user_ratings = pd.read_csv('user_ratings.csv', index_col=0)


def recommend_neighborhood(new_user_query, model, k=10, mean_input=True, ratings=user_ratings):
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
    neighborhood_filter = neighborhood.drop(new_user_query.keys(), axis=1)
    # calculate their average rating

    df_score = neighborhood_filter.sum()

    df_score_ranked = df_score.sort_values(ascending=False).reset_index()

    recommendations = df_score_ranked[:k]
    # filter out movies already seen by the user
    recommendations.columns = ['Title', 'Rating']
    # return the top-k highest
    return recommendations


def recommend_nmf(new_user_query, model,  k=10, mean_input=True, ratings=user_ratings):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model.
    Returns a list of k movie ids.
    """

    # Candidate generation
    new_user_dataframe = pd.DataFrame(new_user_query, columns=movies, index=['new_user'])
    # Construct a user vector
    if mean_input:
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
    # return the top-k the highest rated movie ids or titles
    ranked_f = R_hat_new_user_filter.T.sort_values(by=['new_user'], ascending=False)

    recommendation = ranked_f[:k].reset_index()
    recommendation.columns = ['Title', 'Rating']
    return recommendation

