# Persona Description based on LSA topic model
## Data Description
- 包含8000多个微博⽤用户id及其标签列表、标签的权重分数
- 每个⼈人的标签列表包含通过某种算法⽣生成的20个标签，该算
法也产⽣生了标签权重，刻画了标签对⽤用户的特征描述能力

## Task Description
- 利⽤用LSA、LDA等主题模型并结合KNN、K-Means等分/聚类算法将这些⽤用户进⾏行分组
- 对每组⽤用户进⾏行群体画像. 可基于TF-IDF等标签权重计算⽅方法为每组⽤用户选出最具代表性的标
签集合
