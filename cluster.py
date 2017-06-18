from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl

from mpl_toolkits.mplot3d import Axes3D
# from pylab import *

EPS = 5
SAMPLES = 5
METRIC = "euclidean"
NUM_CLUSTERS = 5

def cluster(alg, path, cols, start_year=None, players=None, show_label=True):
	# read in and clean up the data
	data = read_csv(path)
	if start_year:
		data = data.loc[data["Year"] > start_year]
	data = data.fillna(0)

	if players:
		data = data.loc[data["Player"].isin(players)]

	# for labeling later
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = ["%s %.0f" % pair for pair in zip(names, years)]

	data = data[cols]
	data = data.as_matrix().astype("float32", copy=False)

	# use the indicated clustering algorithm
	alg = alg.lower()
	if alg == "kmeans":
		clusterer = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
	elif alg == "dbscan":
		clusterer = DBSCAN(eps=EPS, metric=METRIC, min_samples=SAMPLES)
	else:
		raise ValueError("Clustering algorithm unrecognized.")

	clusterer.fit(data)

	# reduce dimensions for visualization
	pca = PCA(n_components=2).fit(data)
	pca_2d = pca.transform(data)

	# plot
	# pl.figure('K-Means')

	unique_labels = set(clusterer.labels_)

	if show_label:
		i = 0
		for unique_label in unique_labels:
			pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=clusterer.labels_)
			for name_year, x, y in zip(names_years, pca_2d[:, 0], pca_2d[:, 1]):			
				if clusterer.labels_[i] == unique_label:
					pl.annotate(
						name_year,
						xy=(x, y), xytext=(-3, 3), fontsize=6,
						textcoords='offset points', ha='right', va='bottom',
						bbox=dict(boxstyle='round, pad=0.2', fc="skyblue", alpha=0.5),
						arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
				i += 1
			i = 0
			pl.title("NBA Players Stats from %d Clustered by %s using %s"
				% (start_year+1, ", ".join(cols), alg))
			pl.show()
	# pl.show()

def db_cluster(path, cols, start_year=None, players=None, show_label=True):
	# read in and clean up the data
	data = read_csv(path)
	if start_year:
		data = data.loc[data["Year"] > start_year]
	data = data.fillna(0)

	if players:
		data = data.loc[data["Player"].isin(players)]

	# for labeling later
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = ["%s %.0f" % pair for pair in zip(names, years)]

	data = data[cols]
	data = data.as_matrix().astype("float32", copy=False)

	# normalize
	# stscaler = StandardScaler().fit(data)
	# data = stscaler.transform(data)

	print ("Before the fit function")
	print (data)
	print (data.shape)

	# cluster
	db = DBSCAN(eps=EPS, metric='euclidean', min_samples=SAMPLES)
	db.fit(data)
	
	print (Counter(db.labels_))
	
	labels = db.labels_

	# number of clusters in labels, ignoring noise if present
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print("Number of clusters: %d" % n_clusters_)
	# print("Silhouette Coefficient: %0.3f"
	#       % metrics.silhouette_score(data, labels))

	print ("Before the function call")
	print (data)
	print (data.shape)

	plot_clusters(data, db, names_years,
				"NBA Players Stats Clustered by: " + ", ".join(cols),
				show_label, color_by=players)

def db_cluster_mvp(season_path, mvp_path, cols, show_label=True):
	season_data = read_csv("./Data/season_stats.csv")
	mvp_data = read_csv("./Data/mvp.csv")
	data = pd.merge(mvp_data, season_data, how="inner", left_on=["Year", "Player"], right_on=["Year", "Player"])
	
	data = data.fillna(0)

	# for labeling later
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = ["%s %.0f" % pair for pair in zip(names, years)]

	data = data[cols]
	data = data.as_matrix().astype("float32", copy=False)

	# normalize
	# stscaler = StandardScaler().fit(data)
	# data = stscaler.transform(data)

	# cluster
	db = DBSCAN(eps=100, metric='euclidean', min_samples=2)
	db.fit(data)
	
	print (Counter(db.labels_))
	
	labels = db.labels_

	# number of clusters in labels, ignoring noise if present
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print("Number of clusters: %d" % n_clusters_)
	print("Silhouette Coefficient: %0.3f"
		  % metrics.silhouette_score(data, labels))

	plot_clusters(data, db, names_years, "NBA MVP Stats Clustered by: " + ", ".join(cols), show_label)

# TODO: explore correlations
# TODO: check against player positions
def kmeans_cluster(path, cols, start_year=None, players=None, show_label=True):
	# read in and clean up the data
	data = read_csv(path)
	if start_year:
		data = data.loc[data["Year"] > start_year]
	data = data.fillna(0)

	if players:
		data = data.loc[data["Player"].isin(players)]

	# for labeling later
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = ["%s %.0f" % pair for pair in zip(names, years)]

	data = data[cols]
	data = data.as_matrix().astype("float32", copy=False)

	kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
	kmeans.fit(data)

	# reduce dimensions for visualization
	pca = PCA(n_components=2).fit(data)
	pca_2d = pca.transform(data)

	# plot
	pl.figure('K-Means')
	pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)

	if show_label:
		i = 0
		for name_year, x, y in zip(names_years, pca_2d[:, 0], pca_2d[:, 1]):
			if kmeans.labels_[i] == 3:
				pl.annotate(
					name_year,
					xy=(x, y), xytext=(-3, 3), fontsize=6,
					textcoords='offset points', ha='right', va='bottom',
					bbox=dict(boxstyle='round, pad=0.2', fc="skyblue", alpha=0.5),
					arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
			i += 1
	pl.title("NBA Players Stats from %d on Clustered by %s" % (start_year+1, ", ".join(cols)))
	pl.show()


# TODO: use PCA
def plot_clusters(data, db, annotations, title, show_label, color_by):
	"""Function for plotting DBSCAN clusters. Largely taken from:
	http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
	"""
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True

	labels = db.labels_
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

	print ("Before the for loop")
	print (data)
	print (data.shape)
	
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# use black for noise
			col = 'k'

		class_member_mask = (labels == k)

		xy = data[class_member_mask & core_samples_mask]
		
		print ("Inside the for loop")
		print (data)
		print (data.shape)

		print ("First xy")
		print (xy)

		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)

		xy = data[class_member_mask]

		print ("Second xy")
		print (xy)

		plt.plot(xy[:, 3], xy[:, 4], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=6)

	# plt.cm.Spectral(np.linspace(0, 1, len(color_by)))
	if show_label:
		# some serious hack going on to make colors pretty
		if color_by:
			colors = plt.cm.Spectral(np.linspace(0, 1, len(color_by)))
		for annotation, x, y in zip(annotations, data[:, 0], data[:, 1]):
			# assuming elements of color_by are of the form "[player first name] [player last name]"
			# and that annotation is of the form "[player first name] [player last name] [year]"
			player_name = annotation.split(" ")[0] + " " + annotation.split(" ")[1]
			plt.annotate(
				annotation,
				xy=(x, y), xytext=(-3, 3), fontsize=6,
				textcoords='offset points', ha='right', va='bottom',
				bbox=dict(boxstyle='round, pad=0.2', fc=colors[color_by.index(player_name)], alpha=0.5),
				arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))

	plt.title(title)
	plt.show()

if __name__ == "__main__":
	free_throws = ["FT", "FTA", "FT%"]
	threes = ["3P", "3PA", "3P%"]
	all = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%"]
	player_info = ["height", "weight", "college", "born", "birth_city", "birth_state"]

	# path, cols, start_year=None, players=None, show_label=True
	# kmeans_cluster("./Data/season_stats.csv", free_throws, start_year=2016, show_label=True)
	cluster("kmeans", "./Data/season_stats.csv", free_throws, start_year=2016, show_label=True)
	# db_cluster_mvp("./Data/season_stats.csv", "./Data/mvp.csv", all)
	# db_cluster("./Data/season_stats.csv", all, players=["LeBron James", "Kobe Bryant", "Stephen Curry"])