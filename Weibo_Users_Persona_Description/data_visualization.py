from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data_pca_tsne = TSNE(n_components=2).fit_transform(data_pca)

plt.scatter(data_pca_tsne[:,0],data_pca_tsne[:,1],s=0.5)
plt.show()
