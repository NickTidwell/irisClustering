
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import datasets
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


digits = datasets.load_digits()

scaler=StandardScaler()
X_scaled = scaler.fit_transform(digits.data)

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_pca)
labels = kmeans.predict(X_pca)
centroids = kmeans.cluster_centers_
colors = ['red', 'blue', 'yellow', 'black', 'green', 'cyan', 'magenta', 'maroon', 'navy', 'orange']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
Xax=X_pca[:,0]
Yax=X_pca[:,1]

plt.scatter(Xax, Yax, c=labels, s=50, cmap=cmap)
#plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#050505', s=1000)

plt.show()


