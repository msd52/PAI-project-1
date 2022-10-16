import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans

train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)

n_clusters = 10
kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 42)
kmeans.fit(train_features)


centers = kmeans.cluster_centers_
labels = kmeans.labels_

min_dists = [np.inf for i in range(len(centers))]
indices = [-1 for i in range(len(centers))]

for idx in range(len(train_features)):
    feat = train_features[idx]
    label = labels[idx]
    dist = np.sqrt((feat[0] - centers[label][0])**2 + (feat[1] - centers[label][1])**2)

    if dist < min_dists[label]:
        min_dists[label] = dist
        indices[label] = idx

new_feats = []
new_gds = []
for idx in range(len(indices)):
    index = indices[idx]
    if index == -1:
        continue
    new_feats.append(train_features[index])
    new_gds.append(train_GT[index])


new_feats = np.array(new_feats)
new_gds = np.array(new_gds)
