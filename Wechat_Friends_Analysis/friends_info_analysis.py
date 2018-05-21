import pickle
from pandas import DataFrame
import re
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import PIL.Image as Image


def get_var(var):
    variable = {}
    for i in friends:
        value = i[var]
        try:
            variable[value] += 1
        except KeyError:
            variable[value] = 1
    return variable


def sex_analysis(friends):
    male = female = other = 0
    for i in friends[1:]:
        sex = i["Sex"]
        if sex == 1:
            male += 1
        elif sex == 2:
            female += 1
        else:
            other += 1
    return male, female, other


def getFriendsData():
    NickName = get_var("NickName")
    Sex = get_var('Sex')
    Province = get_var('Province')
    City = get_var('City')
    Signature = get_var('Signature')
    data = {'NickName': NickName, 'Sex': Sex, 'Province': Province,
            'City': City, 'Signature': Signature}
    return data


def getSignatureText():
    siglist = []
    for i in friends:
        signature = i["Signature"].strip().replace(
            "span", "").replace("class", "").replace("emoji", "").strip()
        if signature:
            rep = re.compile("1f\d+\w*|[<>/=]")
            signature = rep.sub("", signature)
            siglist.append(signature)
    text = "".join(siglist)
    return text


def getSignatureWordCloud(text):
    word_cut = jieba.cut(text, cut_all=True)
    wordlist = []
    for word in word_cut:
        if word:
            wordlist += [word]
    word_space_split = " ".join(wordlist)

    coloring = np.array(Image.open("wechat-logo.png"))
    my_wordcloud = WordCloud(background_color='black', max_words=1000,
                             mask=coloring, max_font_size=60, random_state=42,
                             scale=0.5, min_font_size = 10,
                             font_path="msyh.ttc").generate(word_space_split)
    image_colors = ImageColorGenerator(coloring)
    plt.imshow(my_wordcloud.recolor(color_func=image_colors))
    plt.imshow(my_wordcloud)
    plt.axis("off")
    foo_fig = plt.gcf() 
    foo_fig.savefig("wechat_test2.png",format='png',dpi=300)
    plt.show()  
    plt.close()
    return 0

with open('friends_info.pkl', 'rb') as file:
    friends = pickle.load(file)

male, female, other = sex_analysis(friends)

data = getFriendsData()
frame = DataFrame(data)

text = getSignatureText()
getSignatureWordCloud(text)
