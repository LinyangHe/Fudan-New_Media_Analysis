import numpy as np

# word2vec_dic = pandas.read_table(
#     'D:/NLP/Data/word2vec/sgns.financial.bigram', encoding='latin-1', sep=' ')
# # word2vec_dic = pandas.read_table('test.txt',sep=' ')
# # print(word2vec_dic)

def main():
	word2vec_dic = {}
	with open('D:/NLP/Data/word2vec/sgns.financial.bigram', encoding='utf-8', errors='ignore') as file:
		file.readline()
		for line in file:
			if not line:
				continue
			line = line.strip().split(' ')
			word2vec_dic[line[0]] = np.array([float(i) for i in line[1:]])
	return word2vec_dic

if __name__ == '__main__':
	word2vec_dic = main()