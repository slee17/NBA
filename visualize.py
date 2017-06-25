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
	years = np.arange(start_year, end_year+1, 1)

	print (pd.concat(g for _, g in player_data.groupby("Player") if len(g) > 1))

	# convert to age if the given category is "born"
	if category == "born":
		data[category] = data["Year"].sub(data[category], axis=0)

	# print (data.loc[data["born"] > 60])

	data = data.groupby(by="Year")
	
	# get averages
	means = data.mean()
	medians = data.median()

	fig, ax = plt.subplots()

	colors = ["lightseagreen", "orange"]
	for year in years:
		y = data.get_group(year)[category].tolist()
		ax.scatter([year] * len(y), y, c=colors[year%2], marker="x")

	# plot lines
	line1, = ax.plot(years, means, label="Means", c="crimson")
	ax.legend(loc="upper right", fontsize=10, fancybox=True, framealpha=0.1, borderaxespad=1)

	plt.xticks(np.arange(start_year, end_year+1, 5))
	plt.title("Visualization of NBA Players %s" % category.title())
	plt.show()

if __name__=='__main__':
	visualize_size("./Data/Players_original.csv", "./Data/season_stats.csv", "born")