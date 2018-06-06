## <center>微博用户数据分析</center>

<p style="text-align:right">姓名：左文轩</p>
<p style="text-align:right">学号：15307130170</p>

### 1. 用户分组

#### 1.1 利用LDA主题模型进行分析

由于算法是上一次作业的内容，其基本原理在此不再赘述。这里使用 ```gensim``` 中的 LDA 模型，将每一个用户的标签视为文章，将所有用户的标签视为文档进行主题提取，所得到的效果如下。从结果中可以看出前十个主题中，由于'旅游', '摄影', '电影', '音乐', '文学', '美食', '时尚', '自由'这些标签被过多用户选择（音乐、电影、美食的用户超过2/3），极大地影响了主题模型的训练效果，所以这些标签在多个主题中出现。其原因是在微博用户注册时，这些类是很多用户的共同选择，或者说是默认选择，即关注科技领域的用户会选择这些标签，同时关注文学领域的用户也会有很大部分选择这些标签。从而在主题提取时这些标签与一些其他领域的标签混淆在一起。但是仍然会有其他标签有较好的效果，例如 topic 5 中所涵盖的主题都是与宗教有关的、topic 3 则与娱乐圈有关、 topic 8 中则部分与经济领域有关。

```python
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

dictionary = Dictionary(newtag_list)
corpus = [dictionary.doc2bow(text) for text in newtag_list]
lda = LdaModel(corpus=corpus, id2word=dictionary, alpha=0.2, num_topics=10, random_state=123)
for t, components in lda.print_topics(num_words=7):
    print("topic %d"%t)
    print(components)
    print("---"*20)
```

![LDAtopics](E:\University\grade3sec\broadcast\hw3\LDAtopics.png)



### 2. 用户群体画像

#### 2.1 基于 Word2vec 选取具有代表性的标签集合

（1）算法介绍

word2vec 模型为了改善自然语言的词袋、n-gram表示等传统模型特征维度过高、语义信息残缺等劣势而提出。word2vec本质是含有一个隐藏层的神经网络，分别有 CBOW 和 skip-gram 两种方法来优化该网络，其中 CBOW 是利用周围词的向量表示来预测当前词的向量表示；skip-gram 则是用当前词来预测周围语义。在本次作业中，使用了已经训练好的词向量模型（基于120G语料，包含百度百科、搜狐新闻、小说，由于词向量模型太大，没有上传，[该链接](https://weibo.com/p/23041816d74e01f0102x77v)可以获取模型）。得到每一个用户的标签矩阵，与其权重的线性组合的结果作为该用户的特征向量。可视化结果如下图所示：

![userprofile](E:\University\grade3sec\broadcast\hw3\userprofile.png)

从图中可以看出词向量有聚类效果，由于停用了频次过高的标签，所以包含用户节点最多的类别中的标签较为杂乱（如果不停用，效果会好一点），但是为了看出更细致的分类效果，这里给出了停词之后的结果。可以看出“旅行”、“基督徒”、“演员”、“财经”等标签有较好的聚类效果。可以作为具有代表性的标签集合。相关代码如下所示：

```python
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
%matplotlib inline

# 使用预训练好的词向量模型
model = Word2Vec.load('news_12g_baidubaike_20g_novel_90g_embedding_64.model')
# 获取用户特征向量，是每一个标签的词向量的线性组合，组合系数是所给的标签权重
tag_veclist = []
for i in range(len(newtag_list)):
    wordveclist = []
    for tag in newtag_list[i][1:]:
        if tag not in model:
            wordveclist.append(np.zeros(100))
        else:
            wordveclist.append(model[tag])
    tag_veclist.append(newtag_score[i].dot(np.array(wordveclist)))
tsne = TSNE(2)
decomp_result = tsne.fit_transform(tag_veclist)

mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
fig, ax = plt.subplots(1, 2, figsize = [16, 8])
ax[0].scatter(decomp_result[:, 0], decomp_result[:, 1], s = 15, c = 'blue', edgecolors='black')
ax[1].scatter(decomp_result[:, 0], decomp_result[:, 1], s = 15, c = 'blue', edgecolors='black')
# 停用频次过高的标签
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
```

