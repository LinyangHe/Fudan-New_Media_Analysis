import pandas as pd
import os
import json
import numpy as np
import sqlite3
import jieba

def load_database(db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    # There's a table named instant_news in the database file.
    cur.execute("select * from instant_news;")
    news = pd.read_sql_query("select * from instant_news;", conn)
    return news

news = load_database(
    'C:/Users/Leon/OneDrive/DataScience/10_New Media/Project/Data/opinion.db')
news_head = news.head()
