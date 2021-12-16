from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves
import seaborn as sns
from Modeling.dimensionality_reduction import *

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
from Preprocessing.d_data_encoding import *
from Preprocessing.eda import *
from Modeling.dimensionality_reduction import *
from Modeling.clustering import *


def process_artists() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 4 columns:

    """
    df_artists = read_dataset(Path('Datasets\\artist.csv'))

    print(df_artists.describe())

    # preprocessing on artists and normalize some of the columns
    temp = df_artists.copy()
    clmn = ['duration_ms', 'loudness', 'tempo', 'key', 'count']
    for c in clmn:
        temp[c] = normalize_column(temp[c])
    temp = temp.drop(['artists', 'mode'], axis=1)

    return temp


def process_songs() -> pd.DataFrame:
    """
    Consider the example above and once again perform a preprocessing and cleaning of the iris dataset.
    This time, use normalization for the numeric columns and use label_encoder for the categorical column.
    """
    df_songs = read_dataset(Path('Datasets\\songs.csv'))

    print(df_songs.describe())
    # lets drop songs with popularity 0 so that we just cluster popular songs
    df2 = df_songs[df_songs['popularity'] > 0]
    df2.reset_index(inplace=True)

    # for the clustering part we just focus on features that are related to characteristics of songs, so drop other columns
    df1 = df2.drop(
        ['artists', 'duration_ms', 'explicit', 'key', 'mode', 'release_date', 'year'],
        axis=1)

    df1['loudness'] = normalize_column(df1['loudness'])
    df1['tempo'] = normalize_column(df1['tempo'])
    df1['popularity'] = normalize_column(df1['popularity'])
    # df1['key'] = normalize_column(df1['key'])

    return df1





#plotting learning curves -

def plot_model_learning_curves(xtrain, ytrain, xtest, ytest,model ,scoring_method):
    plot_learning_curves(xtrain, ytrain, xtest, ytest, model,scoring=scoring_method)
    plt.show()

# plot_model_learning_curves(X_train, ypred, X_test, ypred_test, rf_cv_grid.best_estimator_, 'mean_squared_error'):

# feature importance -
def get_feature_importance(dataframe, model):
    ## Get important Features
    feat_importances = pd.Series(model.feature_importances_, index = dataframe .columns)

    ## Sort importances
    feat_importances_ordered = feat_importances.nlargest(n=40)
    # feat_importances_ordered

    ## Plotting Importance
    feat_importances_ordered.plot(kind='barh')
    plt.show()

# get_feature_importance(X_train, clf)

# Elbow curve for clutering
def plot_elbow_curve(dataframe, variables_for_segmenting, max_cluster_no):
    wcss = []

    for i in range(1, max_cluster_no):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(dataframe.loc[:, variables_for_segmenting].values)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, max_cluster_no), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


variables_for_segmenting =  ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
# plot_elbow_curve(data_hana, variables_for_segmenting, 20)


def elbow_curve_knn(x,y):
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    error_rate = []

    # Will take some time
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')





#Heatmaps
def heatmaps(dataframe):
    corr = dataframe.corr()
    sns.set(rc={'figure.figsize':(8.7,6.27)})
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))

# heatmaps(songs)
# heatmaps(artists)


# Hana EDA

def pair_plot(dataframe):
    sns.pairplot(dataframe)


def histogram_plot(column):
    sns.histplot(column)


def scatter_plot(x,y,c):
    plt.figure(figsize=(10, 7))
    plt.scatter(x,y, c=c)



def plotting_regression_models():
    # comraing the performance of regression models on artists
    langs = ['Linear Regression', 'Support Vector Machine', 'Random Forest']
    students = [0.4544561920144762 * 100, 0.6205349398540223 * 100, 0.7116880390786268 * 100]
    c = ['palegreen', 'pink', 'violet']
    data = [go.Bar(
        x=langs,
        y=students,
        marker_color=c
    )]
    layout = go.Layout(
        title="Artists' popularity prediction",
        xaxis=dict(
            title="Model"
        ),
        yaxis=dict(
            title="Score"
        ))
    fig = go.Figure(data=data, layout=layout)
    fig