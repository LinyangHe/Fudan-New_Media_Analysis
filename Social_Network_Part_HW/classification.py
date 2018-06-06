from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as KNN
import numpy as np


def global_KMeans(K, user_matrix):
    kmModel = KMeans(n_clusters=K)
    kmModel.fit(user_matrix)
    prediction_personas = kmModel.predict(user_matrix)
    return prediction_personas


def get_personas_count(prediction_personas):
    prediction_counts = {}
    for index, item in enumerate(prediction_personas):
        try:
            # prediction_counts[i] += 1
            prediction_counts[item].append(index)
        except:
            prediction_counts[item] = []
            prediction_counts[item].append(index)
    return prediction_counts

def matrix_process(user_matrix):
    M = user_matrix.shape[0]
    N = user_matrix.shape[1]
    user_matrix_new = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            temp = user_matrix[i][j]
            if temp == temp:
                user_matrix_new[i][j] = temp
            else:
                user_matrix_new[i][j] = 0.0
    return user_matrix_new

def get_words_lists(personas_count_pca, user_df):
    label_list = user_df.columns
    label_num = user_df.shape[1]
    words_lists = []
    for user_lists in personas_count_pca:
        words_list = {}
        for user_index in personas_count_pca[user_lists]:
            for i in range(label_num):
                temp = user_df.iloc[user_index][i]
                if temp:
                    try:
                        words_list[label_list[i]] += 1
                    except:
                        words_list[label_list[i]] = 1
        words_lists.append(words_list)
    return words_lists

# #Model 1 - KMeans without PCA dimensionality reduction(LSA)
# prediction_personas = global_KMeans(20, user_matrix_new)
# personas_count = get_personas_count(prediction_personas)

#Model 2 - KMeans with PCA dimensionality reduction(LSA)
user_matrix_new = matrix_process(user_matrix)
K = 4
data_pca = PCA(n_components=50).fit_transform(user_matrix_new)
prediction_personas_pca = global_KMeans(K, data_pca)
personas_count_pca = get_personas_count(prediction_personas_pca)

#Compute the most influential label TF-IDF
words_lists = get_words_lists(personas_count_pca, user_df)

                 


