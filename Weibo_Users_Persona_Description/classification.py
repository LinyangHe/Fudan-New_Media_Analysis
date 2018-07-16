#If you want to run this file, you have to run user_analysis.py first.
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as KNN
import numpy as np
import math


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

def get_words_lists(personas_count_pca, user_matrix, label_list):
    label_num = user_matrix.shape[1]
    print("????????????????")
    words_lists = []
    for user_lists in personas_count_pca:
        words_list = {}
        for user_index in personas_count_pca[user_lists]:
            print(user_index)
            for i in range(label_num):
                temp = user_matrix[user_index][i]
                if temp:
                    try:
                        words_list[label_list[i]] += 1
                    except:
                        words_list[label_list[i]] = 1
        words_lists.append(words_list)
    return words_lists

def get_TFIDF(words_lists):
    doc_num = len(words_lists)
    IDF = {}
    TFIDF = []

    for user_class in words_lists:
        for temp_label in user_class:
            try:
                IDF[temp_label] += 1
            except:
                IDF[temp_label] = 1

    for user_class in words_lists:
        sum = 0
        for temp_label in user_class:
            sum += user_class[temp_label]
        temp_class = {}
        for temp_label in user_class:
            temp_tf = user_class[temp_label] / sum
            temp_idf = math.log(doc_num/float(IDF[temp_label] + 1))
            temp_class[temp_label] = temp_tf * temp_idf
        TFIDF.append(temp_class)
    return TFIDF

def choose_lab(TFIDF):
    
    return 

# #Model 1 - KMeans without PCA dimensionality reduction(LSA)
# prediction_personas = global_KMeans(20, user_matrix_new)
# personas_count = get_personas_count(prediction_personas)

#Model 2 - KMeans with PCA dimensionality reduction(LSA)
user_matrix = matrix_process(user_matrix)
K = 6
data_pca = PCA(n_components=50).fit_transform(user_matrix)
prediction_personas_pca = global_KMeans(K, data_pca)
personas_count_pca = get_personas_count(prediction_personas_pca)

#Compute the most influential label TF-IDF
label_list = user_df.columns.values.tolist()
words_lists = get_words_lists(personas_count_pca, user_matrix,label_list)
TFIDF = get_TFIDF(words_lists)

                 


