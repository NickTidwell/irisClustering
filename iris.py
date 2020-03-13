import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

digits = datasets.load_digits()

pca=PCA(n_components=2)
X_pca=pca.fit_transform(digits.data)

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_pca)
labels = kmeans.predict(X_pca)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red', 'blue', 'yellow', 'black', 'green', 'cyan', 'magenta', 'maroon', 'navy', 'orange'])
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=50, cmap=cmap)
plt.show()


