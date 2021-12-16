import pandas as pd
import numpy as np
from pathlib import Path
from Preprocessing.a_load_file import read_dataset
from scipy.spatial import distance


def recommendation(clustered_df, song_id, no_of_songs):

    clustered_df2 = clustered_df[clustered_df['id'] == song_id][
        ['acousticness', 'danceability', 'energy', 'instrumentalness',
         'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'kmeans', 'id']]
    song_details = clustered_df2.drop(["kmeans", 'id'], axis=1)
    # print(clustered_df2['kmeans'].values[0])
    song_cluster = clustered_df2['kmeans'].values[0]
    clustered_df3 = clustered_df[clustered_df['kmeans'] == song_cluster][
        ['acousticness', 'danceability', 'energy', 'instrumentalness',
         'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'id']]


    dst = []
    for index, row in clustered_df3.drop(['id'], axis=1).iterrows():
        dst.append(1 - distance.cosine(np.array(row), song_details.values))

    clustered_df3['cos_dis'] = dst

    # top_5 = clustered_df3.sort_values(by = 'kmeans',ascending=False)['id'][:5]
    top_5 = clustered_df3.sort_values(by='cos_dis', ascending=False)[:no_of_songs]

    return top_5

# atao = recommendation(df2,"4dyrqiXUcK29hzrL2elqO3")

# print(atao)