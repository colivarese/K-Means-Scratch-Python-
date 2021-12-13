import numpy as np
from scipy.spatial.distance import cdist
import random as rd


class K_means():

    def __init__(self, k:int, iters:int):
        self.k = k
        self.iters = iters


    def fit(self, data):
        data = np.asarray(data)
        centroids = self.get_random_k_centroids(data)

        for _ in range(self.iters):
            distances = self.get_distance_to_centroids(data, centroids)
            labels = self.assign_to_cluster(distances)
            centroids = self.update_centroids(data, labels) 
        return labels 

        

    def get_random_k_centroids(self, data):
        idx = np.random.choice(len(data), self.k, replace=False)
        centroids = data[idx, :]
        return centroids

    def get_distance_to_centroids(self, data, centroids):
        return cdist(data, centroids ,'euclidean')

    def assign_to_cluster(self, distances):
        return np.array([np.argmin(i) for i in distances])

    def update_centroids(self, data, labels):
        out = np.zeros((self.k, data.shape[1]))
        centroids = {key:0 for key in range(self.k)}
        for val, idx in zip(data, labels):
            out[idx] = (out[idx] + val) / 2
        return out

    def predict(self, true_labels, predictions):
        total = 0
        for true, pred in zip(true_labels, predictions):
            if true == pred:
                total += 1
        print(f'Accuracy is: {total / len(predictions)}')