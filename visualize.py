from pandas import read_csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_size(player_path, season_path, category):
	# read in and clean up the data
	season_data = read_csv(season_path)
	player_data = read_csv(player_path)
	data = pd.merge(season_data, player_data, how="left", on=["Player"])
	
	# keep only columns of interest
	data = data[["Year", "Player", category]]

	# clean up
	data = data.dropna()

	start_year = int(data["Year"].min())
	end_year = int(data["Year"].max())

	data = data.groupby(by="Year")
	
	# get averages
	means = data.mean()
	medians = data.median()

	colors = ["lightseagreen", "orange"]
	# x = [year for year in range(start_year, end_year+1)]
	for year in range(start_year, end_year+1):
		y = data.get_group(year)[category].tolist()
		plt.scatter([year] * len(y), y, c=colors[year%2], marker="x")

	plt.xticks(np.arange(start_year, end_year+1, 5))
	plt.show()

if __name__=='__main__':
	visualize_size("./Data/players.csv", "./Data/season_stats.csv", "height")