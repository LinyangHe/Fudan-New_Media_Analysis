#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import zeros
from numpy import log
import numpy
from scipy.linalg import svd
#from math import sum

titles=[
 "The Neatest Little Guide to Stock Market Investing",
 "Investing For Dummies, 4th Edition",
 "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
 "The Little Book of Value Investing",
 "Value Investing: From Graham to Buffett and Beyond",
 "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
 "Investing in Real Estate, 5th Edition",
 "Stock Investing For Dummies",
 "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
 ]
stopwords = ['and','edition','for','in','little','of','the','to']
ignorechars = ''',:'!'''

class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0

    def parse(self, doc):
        words = doc.split()
        for w in words:
            w = w.lower().strip(ignorechars)

            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d] += 1

    def TFIDF(self):
        WordsPerDoc = self.A.sum(axis=0)
        DocsPerWord = numpy.asarray(self.A > 0, 'i').sum(axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])

    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)
    
    def printA(self):
        print (self.A)
        return self.A
    def printU(self):
        print (self.U)
        return self.U
    def printS(self):
        print (self.S)
        return self.S
    def printV(self):
        print (self.Vt)
        return self.Vt


mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)

mylsa.build()
A = mylsa.printA()
mylsa.TFIDF()
mylsa.calc()
U = mylsa.printU()
S = mylsa.printS()
V = mylsa.printV()
w_dict = mylsa.wdict
d_count = mylsa.dcount