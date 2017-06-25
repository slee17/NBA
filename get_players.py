"""Original code from: https://www.kaggle.com/drgilermo/nba-players-stats/discussion/35227"""

import pandas as pd
import time
import re
import requests

from bs4 import BeautifulSoup

def get_players():
	df = pd.read_csv("./Data/season_stats.csv")
	players_df = pd.DataFrame()
	players_df['Player'] = df.Player.unique()
	t = time.time()
	regex = re.compile('[^a-zA-Z]')
	for j, player in enumerate(players_df['Player']):
		try:
			college = ''
			birth_place = ''
			draft_year = ''
			draft_pick = ''
			draft_team = ''
			name = player.split(' ')[0].lower()
			surname = player.split(' ')[1].lower()
			first_letter = surname[0].lower()
			surname = regex.sub('',surname)
			path ="http://www.basketball-reference.com/players/' + str(first_letter) + '/' + surname[:5] + name[:2] + '01.html"
			r = requests.get(path, headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36"})
			soup = BeautifulSoup(r.content)
			# print (j, 'of....', len(players_df))

			for i, tag in enumerate(soup.find_all('p')):
				if i in range(20):
					t = tag
					print (t.text.split('\n'))

					try:
						if t.text.split('\n')[2] == u'  Draft:':
							draft_year =  t.text.split('\n')[4].split(',')[-1][:5]
							draft_team =  t.text.split('\n')[4].split(',')[0]
							if len(t.text.split('\n')[4].split(',')) == 3:
								draft_pick =  t.text.split('\n')[4].split(',')[1]
							else:
								draft_pick = ''
					except IndexError:
						pass

					try:
						if t.text.split('\n')[0][1] == '-':
							weight = float(t.text.split('\n')[0].split('(')[1].split(',')[1].split('k')[0])
							height = float(t.text.split('\n')[0].split('(')[1].split(',')[0].split('cm')[0])
							players_df.loc[j,'weight'] = weight
					except IndexError:
						pass

					try:                                 
						if t.text.split('\n')[2] ==  u'  College:':
							college = t.text.split('\n')[4]
					except IndexError:
						pass

					try:
						if t.text.split('\n')[1] == 'Born:':
							born =  float(t.text.split('\n')[4])
							birth_place = t.text.split('\n')[7]
							players_df.loc[j,'born'] = born
							players_df.loc[j,'birth_place'] = birth_place
					except IndexError:
						pass

			# print (player.split(' ')[0].lower(),college,weight,height,born, birth_place, draft_team,draft_year)
			# players_df.loc[j,'heig(ht '] = heig ht)

			# players_df.loc[j,'weight'] = weight
			players_df.loc[j,'college'] = college
			# players_df.loc[j,'born'] = born
			# players_df.loc[j,'birth_place'] = birth_place
			# players_df.loc[j,'Draft_year'] = draft_year
			players_df.loc[j,'Draft_team'] = draft_team
			players_df.loc[j,'Draft_pick'] = draft_pick

		except AttributeError:
			pass

		players_df.to_csv("./Data/players_new.csv")

if __name__ == "__main__":
	get_players()