from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	# data
	data = read_csv('Seasons_Stats.csv')
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = []
	for i in range(len(names)):
		names_years.append("%s %f" % (names[i], years[i]))
	# only keep stats after 1999
	data = data.loc[data["Year"] > 2015]
	
	# drop non-continuous variables
	data.drop(["Unnamed: 0", "Year", "Player", "Pos", "Tm"], axis=1, inplace=True)
	data = data.fillna(0)
	data = data[["3P", "3PA", "3P%"]]
	data = data.as_matrix().astype("float32", copy=False)
	# normalize
	# stscaler = StandardScaler().fit(data)
	# data = stscaler.transform(data)

	# cluster
	db = DBSCAN(eps=3, metric='euclidean', min_samples=2)
	db.fit(data)
	print (db)

	print (Counter(db.labels_))
	print (db.labels_)

	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)
	# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	# print("Adjusted Rand Index: %0.3f"
	#       % metrics.adjusted_rand_score(labels_true, labels))
	# print("Adjusted Mutual Information: %0.3f"
	#       % metrics.adjusted_mutual_info_score(labels_true, labels))
	# print("Silhouette Coefficient: %0.3f"
	#       % metrics.silhouette_score(X, labels))

	# visualization
	# pca = PCA(n_components=2).fit(data)
	# pca_2d = pca.transform(data)
	# for i in range(0, pca_2d.shape[0]):
	# 	if db.labels_[i] == 0:
	# 		c1 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
	# 	# elif db.labels_[i] == 1:
	# 	# 	c2 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
	# 	elif db.labels_[i] == -1:
	# 		c3 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
	# plt.legend([c1, c3], ['Cluster 1', 'Noise'])
	# plt.title('DBSCAN Results')
	# plt.show()
	# Black removed and is used for noise instead.
	labels = db.labels_
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = 'k'

	    class_member_mask = (labels == k)

	    xy = data[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=14)

	    xy = data[class_member_mask & ~core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=6)

	for name, x, y in zip(names_years, data[:, 0], data[:, 1]):
	    plt.annotate(
	        name,
	        xy=(x, y), xytext=(-20, 20),
	        textcoords='offset points', ha='right', va='bottom',
	        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
	        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()