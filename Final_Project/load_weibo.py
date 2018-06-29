import pandas
import pymysql
import pickle

def load_database():

    conn = pymysql.connect(host='123.56.23.226', port=3306, user='weibospider',
                           passwd='weibospider!@#', db='weibo', use_unicode=True, charset="utf8")
    weibo_table = pandas.read_sql("select * from weibo.weibo_data limit 10000;", con=conn)
    # users_table = pandas.read_sql("select * from douban.user;", con=conn)
    # return movies_table, users_table
    return weibo_table

weibo_table = load_database()

# with open('./Data/Douban/movies.pkl','wb') as file1:
#     pickle.dump(movies_table, file1)

# with open('./Data/Douban/users.pkl','wb') as file2:
#     pickle.dump(users_table, file2)ump(users_table, file2)