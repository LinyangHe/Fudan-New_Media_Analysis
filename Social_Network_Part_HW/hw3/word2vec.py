# -*- coding:utf-8 -*-
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

# Load the pre-trained word2vec model
model = Word2Vec.load('news_12g_baidubaike_20g_novel_90g_embedding_64.model')
tag_veclist = []

# Compute the user feature vector
for i in range(len(newtag_list)):
    wordveclist = []
    for tag in newtag_list[i][1:]:
        if tag not in model:
            wordveclist.append(np.zeros(100))
        else:
            wordveclist.append(model[tag])
    tag_veclist.append(newtag_score[i].dot(np.array(wordveclist)))

# Use TSNE to reduce dimension to 2D
tsne = TSNE(2)
decomp_result = tsne.fit_transform(tag_veclist)


# Visualization
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
fig, ax = plt.subplots(1, 2, figsize = [16, 8])
ax[0].scatter(decomp_result[:, 0], decomp_result[:, 1], s = 15, c = 'blue', edgecolors='black')
ax[1].scatter(decomp_result[:, 0], decomp_result[:, 1], s = 15, c = 'blue', edgecolors='black')
stop = {'旅游', '摄影', '电影', '音乐', '文学', '美食', '时尚', '自由'}
for i in [np.random.randint(len(newtag_list)) for _ in range(100)]:
    for tagidx in range(1, len(newtag_list[i])):
        if newtag_list[i][tagidx] not in stop:
            break
    ax[1].annotate(newtag_list[i][tagidx], decomp_result[i], rotation=-30.,ha="right", va="top",bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
ax[0].set_title('使用TSNE降维之后的用户向量可视化')
ax[1].set_title('加上标签的结果')
plt.savefig('userprofile.png')
plt.show()
