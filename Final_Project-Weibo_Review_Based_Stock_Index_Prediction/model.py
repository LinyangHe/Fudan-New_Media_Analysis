import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random

def get_data_split(day_num, use_news = True, use_weibo = False):
    if use_news and use_weibo:
        X = np.zeros((day_num,601))
    else:
        X = np.zeros((day_num,301))

    yesterday_rate = 0
    for index, item in enumerate(day_sh_data):
        # print(index, item)
        if use_news and not use_weibo:
            X[index,:300] = day_news_vec[item]
            if day_sh_data[item] > yesterday_rate:
                X[index, 300] = 1
            yesterday_rate = day_sh_data[item]            
            continue

        if not use_news and use_weibo:
            X[index,:300] = day_weibo_vec[item]
            if day_sh_data[item] > yesterday_rate:
                X[index, 300] = 1
            yesterday_rate = day_sh_data[item]                        
            continue
        
        if use_news and use_weibo:
            X[index,:300] = day_news_vec[item]
            X[index,300:600] = day_weibo_vec[item]
            if day_sh_data[item] > yesterday_rate:
                X[index, 600] = 1
            yesterday_rate = day_sh_data[item]            
            
    train_num = int(day_num * 0.8)
    # print(train_num)
    N = 600 if use_news and use_weibo else 300
    print(N)
    X_train = X[:train_num, :N]
    y_train = X[:train_num, N]
    X_test = X[train_num:, :N]
    y_test = X[train_num:, N]

    return X_train, y_train, X_test, y_test

random.seed(10)
X_train, y_train, X_test, y_test = get_data_split(147,use_news=True, use_weibo=False)

# model = KNeighborsClassifier() 
'''
1. Awful when use news and weibo the same time
2. Only use weibo
    Accuracy:  0.5
    Recall:  0.0666666666667
    Precision:  0.5
    F1 Score:  0.117647058824
3. Only use news:
    Accuracy:  0.5
    Recall:  0.2
    Precision:  0.5
    F1 Score:  0.285714285714
'''

# model = LogisticRegression(penalty='l2') 
'''
1. Awful when use news and weibo the same time
2. Awful when use only weibo
2. Awful when use only news data 
'''

# model = RandomForestClassifier(n_estimators=8)
'''
1. When use news and weibo the same time:
    Accuracy:  0.454666666667
    Recall:  0.189
    Precision:  0.395798021015
    F1 Score:  0.242698780774
2. Only use weibo
    Accuracy:  0.471333333333
    Recall:  0.223
    Precision:  0.445474677284
    F1 Score:  0.281671281475
3. Only use news:
    Accuracy:  0.453666666667
    Recall:  0.215666666667
    Precision:  0.393475293862
    F1 Score:  0.265883374265
'''

# model = DecisionTreeClassifier() 
'''
1. When use news and weibo the same time:
    Accuracy:  0.433
    Recall:  0.369
    Precision:  0.42318165471
    F1 Score:  0.392167545952
2. Only use weibo:
    Accuracy:  0.462833333333
    Recall:  0.453333333333
    Precision:  0.464434589662
    F1 Score:  0.453397069787
3. Only use news:
    Accuracy:  0.5215
    Recall:  0.407333333333
    Precision:  0.52985051524
    F1 Score:  0.458112429969
'''

# model = GradientBoostingClassifier(n_estimators=150)
'''
The Best Model!
1. When use news and weibo the same time:
    Accuracy:  0.536166666667
    Recall:  0.205666666667
    Precision:  0.602833333333
    F1 Score:  0.306156641604
2. Only use weibo:
    Accuracy:  0.405333333333
    Recall:  0.214
    Precision:  0.343236652237
    F1 Score:  0.263192925915
3. Only use news:
    Accuracy:  0.48
    Recall:  0.336666666667
    Precision:  0.472947746698
    F1 Score:  0.392683455433
'''

# model = SVC(kernel='rbf', probability=False)
'''
1. Awful when use news and weibo the same time
2. Awful when use weibo only
3. Awful when use news only
'''

model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
'''
1. When use news and weibo the same time:
    Accuracy:  0.498166666667
    Recall:  0.414666666667
    Precision:  0.25615705057
    F1 Score:  0.29705706992
2. Only use weibo:
    Accuracy:  0.508333333333
    Recall:  0.389
    Precision:  0.322593568604
    F1 Score:  0.310424955684
3. Only use news:
    Accuracy:  0.493166666667
    Recall:  0.278333333333
    Precision:  0.186475836855
    F1 Score:  0.207750793704
'''

accuracy_score = []
recall_score = []
precision_score = []
f1_score = []

for i in range(200):
    print(i)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    accuracy_score.append( metrics.accuracy_score(y_test, y_hat))
    recall_score.append( metrics.recall_score(y_test, y_hat))
    precision_score.append( metrics.precision_score(y_test, y_hat))
    f1_score.append( metrics.f1_score(y_test, y_hat))

print("Accuracy: ", sum(accuracy_score)/len(accuracy_score) )
print("Recall: ", sum(recall_score)/len(recall_score))
print("Precision: ", sum(precision_score)/len(precision_score))
print("F1 Score: ",sum(f1_score)/len(f1_score))