#this is the pca for all the data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


def perform_pca(data):
    pca = PCA(n_components=2, svd_solver='full')
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])

    return principalDf

# perform_pca(df1)



def perform_tsne(data):
    tsne = TSNE(n_components=2)
    tsneComponents = tsne.fit_transform(data)
    tsneDf = pd.DataFrame(data=tsneComponents
                          , columns=['tsne 1', 'tsne 2'])
    return tsneDf