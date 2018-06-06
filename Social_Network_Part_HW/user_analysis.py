import pandas as pd


def load_data():
    tag_lists_naive= pd.read_table('./Data/tag_lists.txt', 
    								sep='\s+', header=-1)
    tag_scores = pd.read_table('./Data/tag_scores.txt', sep='\s+', header=-1)
    tag_lists = pd.DataFrame(columns = [i for i in range(21)])
    j = 0
    for i in range(len(tag_lists_naive)):
    	if j % 100 == 0:
    		print(j)
    	temp = tag_lists_naive.iloc[i][1]
    	if temp == temp:
    		tag_lists.loc[j] = tag_lists_naive.iloc[i]
    		j += 1
    return tag_lists, tag_scores

tag_lists, tag_scores = load_data()

