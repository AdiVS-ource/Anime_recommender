import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import NearestNeighbors
df=pd.read_csv('anime.csv')
adf=df.drop( columns=['anime_id','type','members'])
genre_dummies = adf['genre'].str.get_dummies(sep=',')
genre_dummies.columns = genre_dummies.columns.str.strip()
adf = pd.concat([adf, genre_dummies], axis=1)
adf = adf.loc[:, ~adf.columns.duplicated()]

adf=adf.drop(columns=['genre'])
adf = adf.loc[:, ~adf.columns.duplicated()]
adf['rating']=adf['rating'].fillna(adf['rating'].median())

adf=adf.drop_duplicates(subset=['name'])
adf = adf.reset_index(drop=True)
y=adf['name']
x=adf.drop(columns=['name'])
x['episodes'] = x['episodes'].replace('Unknown', np.nan)
x['episodes'] = pd.to_numeric(x['episodes'])
x['episodes']=x['episodes'].fillna(x['episodes'].median())

scaler=MinMaxScaler()
x[['episodes','rating']]=scaler.fit_transform(x[['episodes','rating']])

x['rating']=x['rating'].fillna(x['rating'].median())
x.columns=x.columns.str.strip()
knn=NearestNeighbors(
    n_neighbors=10,
    metric='cosine'
)
knn.fit(x)
anime_indices=pd.Series(adf.index,index=adf['name']).drop_duplicates()
def recommend ( anime_name,n_neighbours=10):
    idx=anime_indices[anime_name]
    distances,indices=knn.kneighbors(x.iloc[[idx]],n_neighbors=n_neighbours+1)
    recommendations=adf.iloc[indices[0][1:]]['name']
    return recommendations

