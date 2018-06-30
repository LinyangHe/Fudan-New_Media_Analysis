import numpy as np
import pandas as pd
import copy
import jieba

def get_day_weibo_data(weibo_data):
    day_weibo_data = {}
    # today = weibo_data.iloc[0]['create_time'].strip().split(' ')[0]
    for i in range(len(weibo_data)):
        time = weibo_data.iloc[i]['create_time']
        day = time.strip().split(' ')[0]
        whole_weibo = weibo_data.iloc[i]['weibo_cont']

        try:
            day_weibo_data[day] += whole_weibo
        except KeyError:
            day_weibo_data[day] = whole_weibo
    
    return day_weibo_data

def get_day_weibo_vec(day_weibo_data):
    day_weibo_vec = {}
    for day in day_weibo_data:
        print(day)
        weibo_split = [i for i in jieba.cut(day_weibo_data[day])]
        word_num = 0
        doc2vec = np.zeros((300))
        for i in weibo_split:
            try:
                doc2vec += word2vec_dic[i]

                word_num += 1
            except KeyError:
                pass
        doc2vec /= word_num
        day_weibo_vec[day] = doc2vec
            
    return day_weibo_vec

# day_weibo_data = get_day_weibo_data(weibo_table)
day_weibo_data_useful = {}
for i in day_sh_data:
    day_weibo_data_useful[i] = day_weibo_data[i]

day_weibo_vec = get_day_weibo_vec(day_weibo_data_useful)