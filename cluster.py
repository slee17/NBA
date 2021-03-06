from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from pandas import read_csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl

# import constants
from constants import *

# TODO: explore correlations
# TODO: check against player positions
def cluster(alg, path, cols, mvp_path=None, start_year=None, players=None, show_label=True):
	# read in and clean up the data
	data = read_csv(path)
	if mvp_path:
		mvp_data = read_csv(mvp_path)
		try:
			data = pd.merge(mvp_data, data, how="inner", left_on=["Year", "Player"], right_on=["Year", "Player"])
		except KeyError:
			data = pd.merge(mvp_data, data, how="inner", left_on=["Player"], right_on=["Player"])
	if start_year:
		data = data.loc[data["Year"] > start_year]
	if players:
		data = data.loc[data["Player"].isin(players)]
		# TODO: throw exception if mvp_path != None and player not an MVP
	# handle rows with nan - could also be implemented by data.fillna(0) but that would
	# contaminate the dataset
	data = data.dropna()

	# for labeling and title later
	names = data["Player"].tolist()
	try:
		years = data["Year"].tolist()
		labels = ["%s %.0f" % pair for pair in zip(names, years)]
	except KeyError:
		labels = names

	players_plotted = "NBA Players"
	if mvp_path:
		players_plotted = "NBA MVPs"
	if players:
		players_plotted = ", ".join(players)

	if not start_year:
		try:
			start_year = data['Year'].min(axis=0)
			title = "%s Stats from %d on Clustered by %s using %s" % (players_plotted, start_year+1, ", ".join(cols), alg.upper())
		except KeyError:
			labels = names
			title = "%s Stats Clustered by %s using %s" % (players_plotted, ", ".join(cols), alg.upper())

	# keep only columns of interest
	data = data[cols]
	data = data.as_matrix().astype("float32", copy=False)

	# normalize
	data = StandardScaler().fit_transform(data)
	
	# use the indicated clustering algorithm
	alg = alg.lower()
	if alg == "kmeans":
		clusterer = KMeans(n_clusters=K_NUM_CLUSTERS, random_state=0)
	elif alg == "dbscan":
		clusterer = DBSCAN(eps=DB_EPS, metric=DB_METRIC, min_samples=DB_SAMPLES)
	elif alg == "spectral":
		clusterer = SpectralClustering(n_clusters=5,
										eigen_solver='arpack',
										affinity="nearest_neighbors")
	elif alg == "agglomerative":
		# connectivity matrix for structured Ward
		connectivity = kneighbors_graph(data, n_neighbors=10, include_self=False)
		# make connectivity symmetric
		connectivity = 0.5 * (connectivity + connectivity.T)
		clusterer = AgglomerativeClustering(n_clusters=5, linkage='ward',
											connectivity=connectivity)
	else:
		raise ValueError("Clustering algorithm unrecognized.")

	clusterer.fit(data)

	# reduce dimensions for visualization
	pca = PCA(n_components=2).fit(data)
	pca_2d = pca.transform(data)

	# plot
	unique_labels = set(clusterer.labels_)

	if show_label:
		i = 0
		for unique_label in unique_labels:
			pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=clusterer.labels_)
			for name_year, x, y in zip(labels, pca_2d[:, 0], pca_2d[:, 1]):            
				if clusterer.labels_[i] == unique_label:
					pl.annotate(
						name_year,
						xy=(x, y), xytext=(-3, 3), fontsize=6,
						textcoords='offset points', ha='right', va='bottom',
						bbox=dict(boxstyle='round, pad=0.2', fc="skyblue", alpha=0.5),
						arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
				i += 1
			i = 0
			pl.title(title)
			pl.show()

if __name__ == "__main__":
	# cluster("dbscan", "./Data/players.csv", ["height", "weight"])
	cluster("dbscan", "./Data/players.csv", ["height", "weight"], mvp_path="Data/mvp.csv")

	# categories = {}
	# categories['free_throws'] = ["FT", "FTA", "FT%"]
	# categories['threes'] = ["3P", "3PA", "3P%"]
	# categories['all'] = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%"]
	
	# for name, category in categories.items():
		# cluster("kmeans", "./Data/season_stats.csv", category, start_year=2016, show_label=True)
		# cluster("kmeans", "./Data/season_stats.csv", category, mvp_path="Data/mvp.csv", show_label=True)
		# cluster("agglomerative", "./Data/season_stats.csv", category, mvp_path="Data/mvp.csv", show_label=True)
		# cluster("dbscan", "./Data/season_stats.csv", category, start_year=2016, show_label=True)