import pandas as pd
import numpy as np


def load_data():
	tag_lists_naive = pd.read_table('./Data/tag_lists.txt',
									sep='\s+', header=-1)
	tag_scores = pd.read_table('./Data/tag_scores.txt', sep='\s+', header=-1)
	tag_lists = pd.DataFrame(columns=[i for i in range(21)])
	j = 0
	for i in range(len(tag_lists_naive)):
		if j % 100 == 0:
			print(j)
		temp = tag_lists_naive.iloc[i][1]
		if temp == temp:
			tag_lists.loc[j] = tag_lists_naive.iloc[i]
			j += 1
	return tag_lists, tag_scores


def create_label_matrix(tag_lists, N, M = 999999):
	user_num = len(tag_lists)
	matrix_naive = {}
	for i in range(user_num):
		for j in range(20):
			try:
				matrix_naive[tag_lists.iloc[i][j + 1]] += 1
			except:
				matrix_naive[tag_lists.iloc[i][j + 1]] = 1

	N = 10
	matrix_processed = {}
	for i in matrix_naive:
		if matrix_naive[i] >= N and matrix_naive[i] <= M:
			matrix_processed[i] = matrix_naive[i]
	return matrix_processed


def create_user_matrix(label_matrix, tag_lists, tag_scores):
	user_num = len(tag_lists)
	label_indices = {}
	index = 0
	for i in label_matrix:
		label_indices[i] = index
		index += 1

	user_matrix = np.zeros((user_num, label_num))
	for i in range(user_num):
		for j in range(20):
			try:
				label_index = label_indices[tag_lists.iloc[i][j+1]]
				user_matrix[i][label_index] = tag_scores.iloc[i][j]
			except:
				pass
	return user_matrix

def get_tag_scores_norm(tag_scores):
	tag_scores_norm = pd.DataFrame(columns=[i for i in range(20)])
	for i in range(8767):
		temp_sum = sum(tag_scores.iloc[i])
		tag_scores_norm.loc[i] = tag_scores.iloc[i]/temp_sum
	return tag_scores_norm

tag_lists, tag_scores = load_data()

tag_scores = get_tag_scores_norm(tag_scores)

label_matrix = create_label_matrix(tag_lists, 10, M=3000)
label_num = len(label_matrix)

user_matrix = create_user_matrix(label_matrix, tag_lists, tag_scores)

user_df = pd.DataFrame(data = user_matrix, columns = [i for i in label_matrix])