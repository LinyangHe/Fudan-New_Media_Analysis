from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as KNN
import numpy as np


def split_data():
    user_matrix_train = np.zeros((7000, 884))
    user_matrix_test = np.zeros((1767, 884))
    user_matrix_train = user_matrix[:7000, :]
    user_matrix_test = user_matrix[7000:, :]
    return user_matrix_train, user_matrix_test


def global_KMeans(K, user_matrix):
    kmModel = KMeans(n_clusters=K)
    kmModel.fit(user_matrix)
    prediction_personas = kmModel.predict(user_matrix)
    return prediction_personas

def global_KNN(K, user_matrix):
    knnModel = KNN(n_neighbors=K)
    knnModel.fit(user_matrix)
    indices = knnModel.kneighbors(user_matrix,return_distance=False)
    return indices

def get_personas_count(prediction_personas):
    prediction_counts = {}
    for i in prediction_personas:
        try:
            prediction_counts[i] += 1
        except:
            prediction_counts[i] = 1
    return prediction_counts

# user_matrix_train, user_matrix_test = split_data()

def matrix_process():
    user_matrix_new = np.zeros((8767, 884))

    for i in range(8767):
        for j in range(884):
            temp = user_matrix[i][j]
            if temp == temp:
                user_matrix_new[i][j] = temp
            else:
                user_matrix_new[i][j] = 0.0
    return user_matrix_new

# #Model 1 - KMeans without PCA dimensionality reduction(LSA)
# prediction_personas = global_KMeans(20, user_matrix_new)
# personas_count = get_personas_count(prediction_personas)

# #Model 2 - KMeans with PCA dimensionality reduction(LSA)
# data_pca = PCA(n_components=50).fit_transform(user_matrix_new)
# prediction_personas_pca = global_KMeans(20, data_pca)
# personas_count_pca = get_personas_count(prediction_personas_pca)

#Model 3 - KNN without PCA dimensionality reduction(LSA)
indices = global_KNN(50, user_matrix_new)


