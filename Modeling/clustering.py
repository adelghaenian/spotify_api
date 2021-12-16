import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


def Simple_K_means(x: pd.DataFrame, n_clusters, score_metric):
    model = KMeans(init='random',n_clusters=n_clusters)
    clusters = model.fit_predict(x)
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def Optics(x: pd.DataFrame) :
    dbs_o = OPTICS(metric= 'euclidean',n_jobs = -1)
# Fitting and predicting the data
    clusters = dbs_o.fit_predict(x)
# Using Silhouette scoring metric to check the efficiency of the cluster. Here too euclidean was only used.
    score = metrics.silhouette_score(x, dbs_o.labels_, metric='euclidean')
    return dict(model=dbs_o, score=score, clusters=clusters)

def Heirarchical_Clustering(dataframe,  n_clusters, affinity, linkage):
    hierarchy_model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    clusters = hierarchy_model.fit_predict(dataframe)
    return clusters

