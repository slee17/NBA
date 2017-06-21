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

# import constants
from constants import *

# TODO: explore correlations
# TODO: check against player positions
def cluster(alg, path, cols, mvp_path=None, start_year=None, players=None, show_label=True):
	# read in and clean up the data
	data = read_csv(path)
	if mvp_path:
		mvp_data = read_csv(mvp_path)
		data = pd.merge(mvp_data, data, how="inner", left_on=["Year", "Player"], right_on=["Year", "Player"])
	if start_year:
		data = data.loc[data["Year"] > start_year]
	if players:
		data = data.loc[data["Player"].isin(players)]
		# TODO: throw exception if mvp_path != None and player not an MVP
	data = data.fillna(0)

	# for labeling later
	years = data["Year"].tolist()
	names = data["Player"].tolist()
	names_years = ["%s %.0f" % pair for pair in zip(names, years)]

	# for title
	if not start_year:
		start_year = data['Year'].min(axis=0)

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
			players_plotted = "NBA Players"
			if mvp_path:
				players_plotted = "NBA MVPs"
			if players:
				players_plotted = ", ".join(players)
			pl.title("%s Stats from %d on Clustered by %s using %s"
				% (players_plotted, start_year+1, ", ".join(cols), alg.upper()))
			pl.show()

if __name__ == "__main__":
	categories = {}
	categories['free_throws'] = ["FT", "FTA", "FT%"]
	categories['threes'] = ["3P", "3PA", "3P%"]
	categories['all'] = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%"]
	# categories['player_info'] = ["height", "weight", "born"]

	for name, category in categories.items():
		# cluster("kmeans", "./Data/season_stats.csv", category, start_year=2016, show_label=True)
		cluster("kmeans", "./Data/season_stats.csv", category, mvp_path="Data/mvp.csv", show_label=True)
		# cluster("dbscan", "./Data/season_stats.csv", category, start_year=2016, show_label=True)
		# db_cluster_mvp("./Data/season_stats.csv", "./Data/mvp.csv", all)