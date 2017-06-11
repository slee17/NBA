from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	data = read_csv('Seasons_Stats.csv')
	
	# clean up
	data = data.loc[data["Year"] > 2016]
	data = data.fillna(0)

	# for labeling later
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = ["%s %.0f" % pair for pair in zip(names, years)]

	# drop non-continuous variables
	data.drop(["Unnamed: 0", "Year", "Player", "Pos", "Tm"], axis=1, inplace=True)
	
	data = data[["3P", "3PA", "3P%"]]
	data = data.as_matrix().astype("float32", copy=False)

	# normalize
	# stscaler = StandardScaler().fit(data)
	# data = stscaler.transform(data)

	# cluster
	db = DBSCAN(eps=10, metric='euclidean', min_samples=10)
	db.fit(data)
	
	print (Counter(db.labels_))
	
	# http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# number of clusters in labels, ignoring noise if present
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print("Number of clusters: %d" % n_clusters_)
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(data, labels))

	# black removed and is used for noise instead
	labels = db.labels_
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # black used for noise
	        col = 'k'

	    class_member_mask = (labels == k)

	    xy = data[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=14)

	    xy = data[class_member_mask & ~core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=6)

	# for name_year, x, y in zip(names_years, data[:, 0], data[:, 1]):
	#     plt.annotate(
	#         name_year,
	#         xy=(x, y), xytext=(-10, 10),
	#         textcoords='offset points', ha='right', va='bottom',
	#         bbox=dict(boxstyle='round, pad=0.5', fc='skyblue', alpha=0.5),
	#         arrowprops=dict(arrowstyle='-', connectionstyle='arc3, rad=0'))

	plt.title('NBA Players Stats')
	plt.show()