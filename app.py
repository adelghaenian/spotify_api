import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from Preprocessing.a_load_file import read_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from typing import List, Dict, Optional

from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt

# from flask_music
from randomsubgroups import RandomSubgroupRegressor

from pathlib import Path
from Preprocessing.a_load_file import read_dataset
from scipy.spatial import distance

from flask import Flask


from Modeling.regression_models import *
from Modeling.Classification import *
from Modeling.recommendation import *

import time
from Preprocessing.d_data_encoding import *
from Preprocessing.eda import *
from Modeling.dimensionality_reduction import *
from Modeling.clustering import *

def prediction_popularity(feature_values, model):
    feature_list = [feature_values]
    popularity_score = model.predict(feature_list)

    return popularity_score


def prediction_genre(feature_values, model, le):
    feature_list = [feature_values]
    le2 = le.inverse_transform(model.predict(feature_list))
    # popularity_score = model.predict(feature_list)
    return le2

from flask import request

df = pd.read_csv(Path('Datasets\clustering_hana.csv'))
df2 = df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'kmeans', 'id']]

rf_song_model = pickle.load(
    open("Saved_models\SVM_songs.sav", 'rb'))

svm_song_model = pickle.load(
    open("Saved_models\\SVM_songs.sav", 'rb'))

lr_song_model = pickle.load(
    open("Saved_models\\LinearRegression_songs.sav", 'rb'))

rf_artist_model = pickle.load(
    open("Saved_models\\rf_regressor_artists.sav", 'rb'))

svm_artist_model = pickle.load(
    open("Saved_models\\SVM_artists.sav", 'rb'))

lr_artist_model = pickle.load(
    open("Saved_models\\LinearRegression_artists.sav", 'rb'))

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/popularity_predict', methods=['GET'])
@cross_origin()
def popularity_predict():
    acousticness = float(request.args.get('acousticness'))
    danceability = float(request.args.get('danceability'))
    duration_ms = float(request.args.get('duration_ms'))
    energy = float(request.args.get('energy'))
    instrumentalness = float(request.args.get('instrumentalness'))
    key = float(request.args.get('key'))
    liveness = float(request.args.get('liveness'))
    loudness = float(request.args.get('loudness'))
    speechiness = float(request.args.get('speechiness'))
    tempo = float(request.args.get('tempo'))
    valence = float(request.args.get('valence'))
    if request.args.get('model') == "rf_model":
        if request.args.get('song_or_artist') == "song":
            model = rf_song_model
        else:
            model = rf_artist_model
    elif request.args.get('model') == "svm_model":
        if request.args.get('song_or_artist') == "song":
            model = svm_song_model
        else:
            model = svm_artist_model
    else:
        if request.args.get('song_or_artist') == "song":
            model = lr_song_model
        else:
            model = lr_artist_model
    if request.args.get('song_or_artist') == "song":
        out = prediction_popularity([acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness,
                                 loudness, speechiness, tempo, valence], model)
    else:
        count = float(request.args.get('count'))
        out = prediction_popularity([acousticness, danceability, duration_ms, energy, instrumentalness, liveness,
                                 loudness, speechiness, tempo, valence,  key, count], model)
    if out:
        return str(out[0])



le = pickle.load(open("Saved_models\\le_genre", 'rb'))
genre_model = pickle.load(
    open("Saved_models\\rf_model_genres", 'rb'))




@app.route('/genre_predict', methods=['GET'])
@cross_origin()
def genre_predict():
    acousticness = float(request.args.get('acousticness'))
    danceability = float(request.args.get('danceability'))
    duration_ms = float(request.args.get('duration_ms'))
    energy = float(request.args.get('energy'))
    instrumentalness = float(request.args.get('instrumentalness'))
    key = float(request.args.get('key'))
    liveness = float(request.args.get('liveness'))
    loudness = float(request.args.get('loudness'))
    mode = float(request.args.get('mode'))
    speechiness = float(request.args.get('speechiness'))
    tempo = float(request.args.get('tempo'))
    valence = float(request.args.get('valence'))
    out_genre = prediction_genre(
        [acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, valence], genre_model,
        le)
    print(str(out_genre[0]))
    return str(out_genre[0])


@app.route('/recommend', methods=['GET'])
@cross_origin()
def recommnd():
    id = request.args.get('id')
    count = request.args.get('count')
    out = recommendation(df2, id, int(count))
    return out.to_json()


@app.route('/')
@cross_origin()
def index():
  return "<h1>Welcome to Spotify API</h1>"

if __name__ == '__main__':
    app.run()
