import pandas as pd
import os
import json
import numpy as np
import sqlite3
import jieba
import sklearn
import copy

def load_database(db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    # There's a table named instant_news in the database file.
    cur.execute("select * from instant_news;")
    news = pd.read_sql_query("select * from instant_news;", conn)
    return news

def load_news():
	news_1 = load_database(
	    'C:/Users/Leon/OneDrive/DataScience/10_New Media/Project/Data/opinion.db')
	news_2 = load_database(
	    'C:/Users/Leon/OneDrive/DataScience/10_New Media/Project/Data/opinion2.db')
	news = pd.concat([news_1, news_2])
	return news

# news = load_news()
news_head = news.head(10000)

# day_news_data = pd.DataFrame(columns=['date','news','word2vec','sh_next_day'])
def get_day_news_data():
    day_news_data = {}
    today = news_head.iloc[0]['time'].strip().split(' ')[0]
    for i in range(len(news_head)):
        time = news_head.iloc[i]['time']
        day = time.strip().split(' ')[0]

        if day > today:
            yesterday = copy.deepcopy(today)
            today = day
            print(yesterday, today)

        closed_time = day + " 15:00:00.000000"
        try:
            whole_news = news_head.iloc[i]['title'] + news_head.iloc[i]['title'] + news_head.iloc[i]['content']
        except:
            whole_news = news_head.iloc[i]['title'] + news_head.iloc[i]['title']        

        try:
            if time > closed_time:
                try:
                    day_news_data[day] += whole_news
                except KeyError:
                    day_news_data[day] = whole_news
            else:
                try:
                    day_news_data[yesterday] += whole_news
                except KeyError:
                    day_news_data[day] = whole_news
        except IndexError:
            pass
    
    return day_news_data

def get_day_news_vec(day_news_data):
    day_news_vec = {}
    for day in day_news_data:
        print(day)
        news_split = [i for i in jieba.cut(day_news_data[day])]
        word_num = 0
        doc2vec = np.zeros((300))
        for i in news_split:
            try:
                doc2vec += word2vec_dic[i]
                # print(doc2vec)
                word_num += 1
            except KeyError:
                pass
        doc2vec /= word_num
        day_news_vec[day] = doc2vec
            
    return day_news_vec, news_split
    
def get_day_sh_data(day_news_data):
    day_sh_data = {}
    sh_data = pd.read_csv('./Data/sh.csv')
    for i in range(len(day_news_data)):
        pass



day_news_data = get_day_news_data()
day_news_vec, new_split = get_day_news_vec(day_news_data)


