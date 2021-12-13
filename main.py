import pandas as pd
from KMeans import K_means
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/gender_classification_v7.csv')
data.gender[(data['gender'] == 'Male')] = 1
data.gender[(data['gender'] == 'Female')] = 0
true_labels = np.asarray(data['gender'])
data.drop('gender', axis=1, inplace=True)

data = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]

kmeans = K_means(3, 200)
predictions = kmeans.fit(data)

colors = ['r','g','b']
plt.figure()
for d, p in zip(data, predictions):
    c = colors[p]
    plt.plot(d[0],d[1], color=colors[p],marker='o', markersize=12)
plt.show()