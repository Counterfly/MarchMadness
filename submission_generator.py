import compute_matchup_probabilities as sub_gen
import os

for season in range(2013, 2017):
  #sub_gen.main(season)
  os.system("python compute_matchup_probabilities.py %d" % season)
  print("Finished season %d" % season)
