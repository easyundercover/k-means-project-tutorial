#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Load data
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
df_raw = pd.read_csv(url)

#Select columns
df = df_raw[['MedInc','Latitude','Longitude']]

#Normalize
scaler = StandardScaler()
df_norm = scaler.fit_transform(df)

#Model
model = KMeans(n_clusters = 2)
model.fit(df_norm)
df2 = scaler.inverse_transform(df_norm)
df2 = pd.DataFrame(df2,columns=['MedInc','Latitude','Longitude'])
df2['Cluster'] = model.labels_

#Convert column 'cluster' to categorical
df2['Cluster']=pd.Categorical(df2['Cluster'])
