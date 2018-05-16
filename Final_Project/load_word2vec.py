import numpy as np

# word2vec_dic = pandas.read_table(
#     'D:/NLP/Data/word2vec/sgns.financial.bigram', encoding='latin-1', sep=' ')
# # word2vec_dic = pandas.read_table('test.txt',sep=' ')
# # print(word2vec_dic)

word2vec_dic = {}
with open('D:/NLP/Data/word2vec/sgns.financial.bigram', encoding='latin-1') as file:
	file.readline()
	for line in file:
		if not line:
			continue
		line = line.strip().split(' ')
		word2vec_dic[line[0]] = np.array([float(i) for i in line[1:]])