#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import zeros
from numpy import log
import numpy as np
from scipy.linalg import svd
import jieba
#from math import sum
# 我们无处安放的青春
# 银河系漫游指南
stopwords = ['的']
ignorechars = ''',:'!，'''


class LSA(object):

    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0
        self.new_dimen = 0

    def parse(self, doc):
        words = [i for i in jieba.cut_for_search(doc, HMM=True)]
        for w in words:
            w = w.lower().strip(ignorechars).strip()
            if not w:
                continue
            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.new_dimen = len(self.keys)
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1

    def TFIDF(self):
        WordsPerDoc = self.A.sum(axis=0)
        DocsPerWord = numpy.asarray(self.A > 0, 'i').sum(axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                if self.A[i, j]:
                    self.A[i, j] = (self.A[i, j] / WordsPerDoc[j]) * \
                        log(float(cols) / DocsPerWord[i])

    def SVD(self):
        self.U, self.S, self.Vt = svd(self.A)

    def getA(self):
        print(self.A)
        return self.A

    def getU(self):
        print(self.U)
        return self.U

    def getS(self):
        print(self.S)
        return self.S

    def getV(self):
        print(self.Vt)
        return self.Vt

    def get_new_dimen(self):
        return self.new_dimen


def make_new_matrix(U, S, V, maintain_dimen):
    m = U.shape[0]
    n = V.shape[0]
    U_new = zeros((m, maintain_dimen))
    S_new = zeros((maintain_dimen, maintain_dimen))
    V_new = zeros((maintain_dimen, n))
    for i in range(maintain_dimen):
        U_new[:, i] = U[:, i]
        S_new[i][i] = S[i]
        V_new[i, :] = V[i, :]

    A_new = np.dot(np.dot(U_new, S_new), V_new)
    return A_new

titles = []
with open('Chinese_titles_romance.txt', encoding='utf-8') as title_file:
    for line in title_file:
        titles.append(line.strip())
titles_num = len(titles)

mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)

mylsa.build()
A = mylsa.getA()
mylsa.TFIDF()
mylsa.SVD()
U = mylsa.getU()
S = mylsa.getS()
V = mylsa.getV()
w_dict = mylsa.wdict
d_count = mylsa.dcount

new_dimen = mylsa.get_new_dimen()
maintain_dimen = 3

A_new = make_new_matrix(U, S, V, maintain_dimen)
