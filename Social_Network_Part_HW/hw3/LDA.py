from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

with open('tag_lists.txt', 'r', encoding='utf-8') as f:
    tag_list = [line.split() for line in f.readlines()]
with open('tag_scores.txt', 'r') as f:
    tag_score = [[float(value) for value in line.split()] for line in f.readlines()]
    
newtag_score = [np.array(tag_score[i]) for i in range(len(tag_list)) if len(tag_list[i])-1 == len(tag_score[i]) and len(tag_list[i]) > 1]
newtag_list = [tag_list[i] for i in range(len(tag_list)) if len(tag_list[i])-1 == len(tag_score[i]) and len(tag_list[i]) > 1]

dictionary = Dictionary(newtag_list)
corpus = [dictionary.doc2bow(text) for text in newtag_list]
lda = LdaModel(corpus=corpus, id2word=dictionary, alpha=0.2, num_topics=10, random_state=123)
for t, components in lda.print_topics(num_words=7):
    print("topic %d"%t)
    print(components)
    print("---"*20)