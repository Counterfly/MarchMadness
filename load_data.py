import pandas as pd
import numpy as np
import os
import sys
import pickle
import scipy.io as scio
from configs import AttributeMapper, DATA_DIRECTORY, CSV_TEAMS, CSV_REGULAR_SEASON_COMPACT, CSV_REGULAR_SEASON_DETAILED, SYMBOL_WIN, SYMBOL_LOSE

class OneHotGenerator:
  def __init__(self, size):
    self.unique_elements = size
    self.current_element = 0

  def next(self):
    assert(self.current_element < self.unique_elements)
    one_hot = np.zeros((self.unique_elements, 1))
    one_hot[self.current_element] = 1
    self.current_element += 1
    return one_hot

class DataSet:
  def __init__():
    pass


def load_teams(force=False):
  one_hot_encoding_file = DATA_DIRECTORY + 'teams_one_hot_encoding.pickle'

  team_to_one_hot = {}
  if force or not os.path.isfile(one_hot_encoding_file):
    # Create One-Hot-Encoding for Team IDs
    df = pd.read_csv(CSV_TEAMS.get_path())
    one_hot_generator = OneHotGenerator(len(df.index))
    for index, row in df.iterrows():
      team_id = int(CSV_TEAMS.get_team_id(row))
      if team_id not in team_to_one_hot: 
        team_to_one_hot[team_id] = one_hot_generator.next().tolist()
    
    # Save one hot encoding so don't have to recompute
    with open(one_hot_encoding_file, 'wb') as out_file:
      pickle.dump(team_to_one_hot, out_file, protocol=pickle.HIGHEST_PROTOCOL)

  else:
    print("using existing one hot encoding for team in %s" % one_hot_encoding_file)

  # Load data and read in as numpy array
  with open(one_hot_encoding_file, 'rb') as team_file:
    team_to_one_hot = pickle.load(team_file)

  for key in team_to_one_hot.keys():
    team_to_one_hot[key] = np.array(team_to_one_hot[key])

  print("team to one hot has size %d" % len(team_to_one_hot))
  return team_to_one_hot


class SeasonData:
  MAX_REGULAR_GAMES_PER_SEASON_PER_TEAM = 40
  def __init__(self, season, team_id, season_attributes):
    '''
      season: year
      team_id: integer
      season_attributes: list of what to keep track of. 
    '''
    self._season = season
    self._team_id = team_id
    self._attributes = {}
    for attribute in season_attributes:
      self._attributes[attribute] = np.zeros((self.MAX_REGULAR_GAMES_PER_SEASON_PER_TEAM, 1))
 
    # wins will be tracked separatly to allow for proper averaging   
    self._wins = np.zeros((self.MAX_REGULAR_GAMES_PER_SEASON_PER_TEAM, 1))


  @property
  def season(self):
    return self._season

  @property
  def team_id(self):
    return self._team_id

  def add_data(self, csvinfo, data_frame, attribute_mapper):
    '''
      Add data based on self._attributes but index the data (data_frame)
      according to attribute_mapper since this will change whether team is winner or loser in a game
      So,
        attr in self._attributes is what we want to use as a feature,
        data_frame is the current game
        attribute_mapper maps attributes in self._attributes to the proper key in data_frame
    '''
    game_number = 0
    while self._wins[game_number] != 0:
      # earliest position of 0 (win is +1, loss is -1)
      game_number += 1
      if game_number >= self.MAX_REGULAR_GAMES_PER_SEASON_PER_TEAM:
        print(self.team_id, self.season)

    for attribute in self._attributes.keys():
      data_key = attribute_mapper[attribute]
      self._attributes[attribute][game_number] = data_frame[data_key]

    is_winner = csvinfo.get_winning_team(data_frame) == self.team_id
    assert(is_winner or csvinfo.get_losing_team(data_frame) == self.team_id)
    self._wins[game_number] = SYMBOL_WIN if is_winner else SYMBOL_LOSE
        
     
  @property
  def wins(self):
    '''1 -> win, -1 -> loss, 0 -> game not played'''
    return self._wins

  @property
  def scores(self):
    '''scores for each game in current season'''
    return self._scores

  def get_attribute(self, attribute):
    '''Retruns the data for the specified attribute or raises KeyError if attribute DNE'''
    return self._attributes[attribute]


class TeamData:
  def __init__(self, teamid, team_one_hot_encoding):
    self.team_id = teamid
    self.team_one_hot = team_one_hot_encoding
    self.seasons = dict()


  def add_data(self, csvinfo, data_frame):
    '''Add data ..which attributes?''' 
    # Create Training data for this game
    is_winner = csvinfo.get_winning_team(data_frame) == self.team_id
    assert(is_winner or csvinfo.get_losing_team(data_frame) == self.team_id)

    season = csvinfo.get_season(data_frame)
    season_data = self.get_or_create_season_data(season)

    attribute_mapper = AttributeMapper.get_map(is_winner)
    season_data.add_data(csvinfo, data_frame, attribute_mapper)

     
  def get_season_win_list(self, csvinfo, data_frame):
    season = csvinfo.get_season(data_frame)
    season_data = self.get_or_create_season_data(season)

    return season_data.get_wins()

  def get_or_create_season_data(self, season):
    if season not in self.seasons:
      self.seasons[season] = SeasonData(season, self.team_id, AttributeMapper.get_attributes()) 
    return self.seasons[season]

  def snapshot(self, season):
    '''Create input example as a column vector'''
    season_data = self.get_or_create_season_data(season)
    season_wins = season_data.wins
    num_games_played = np.count_nonzero(season_wins)

    attribute_list = []
    attributes = sorted(AttributeMapper.get_map(True).keys())
    for attribute in attributes:
      attribute_list.append(np.sum(season_data.get_attribute(attribute)) / num_games_played)

    attribute_column = np.array(attribute_list).reshape(len(attribute_list), 1)
    snap = np.concatenate((self.team_one_hot.reshape(-1,1), season_wins.reshape(-1,1), attribute_column), axis=0)

    return snap

    

def load_regular_season_games(detailed=False, NUM_HISTORIC_WIN_LOSS=10):
  def get_historic_win_loss(team1, team2, num_previous_games):
    '''
      Find the num_previous_games most recent games team1 and team2 have played against each other
      and return a vector of the win-loss wrt team1
    '''
    return np.zeros((num_previous_games, 1))

  if detailed:
    CSV_GAMES = CSV_REGULAR_SEASON_COMPACT
  else:
    CSV_GAMES = CSV_REGULAR_SEASON_DETAILED


  team_to_one_hot = load_teams()
  team_data = {}

  LAST_SEASON=2017

  # dictionaries of season->input examples
  _input = dict()
  _output = dict()
  with open(CSV_GAMES.get_path()) as games_file:
    df = pd.read_csv(games_file)
    print("data size = %d" %(len(df.index)))
    df = df.loc[df[CSV_GAMES.SEASON] <= LAST_SEASON]
    print("new data size = %d" %(len(df.index)))
    iteration_count = 0
    for _, row in df.iterrows():
      winning_team = CSV_GAMES.get_winning_team(row)
      losing_team = CSV_GAMES.get_losing_team(row)

      season = CSV_GAMES.get_season(row)
      
      iteration_count += 1
      if iteration_count % 3000 == 0:
        print("iteration = %d, season=%d" % (iteration_count, season))
        sys.stdout.flush()


      if winning_team not in team_data:
        '''Create initial empty season data'''
        team_data[winning_team] = TeamData(winning_team, team_to_one_hot[winning_team])

      if losing_team not in team_data:
        '''Create initial empty season data'''
        team_data[losing_team] = TeamData(losing_team, team_to_one_hot[losing_team])


      if season not in _input:
        _input[season] = list()
        _output[season] = list()

      # Create two training examples for symmetry
      wteam_snapshot = team_data[winning_team].snapshot(season)
      lteam_snapshot = team_data[losing_team].snapshot(season)
      historic_win_loss = get_historic_win_loss(winning_team, losing_team, NUM_HISTORIC_WIN_LOSS)

      train = np.vstack((wteam_snapshot, lteam_snapshot, historic_win_loss.reshape(-1,1)))
      _input[season].append(train.T)

      assert(SYMBOL_WIN == 1)
      assert(SYMBOL_LOSE == -1)
      historic_win_loss = np.multiply(-1, historic_win_loss)
      train = np.concatenate((lteam_snapshot, wteam_snapshot, historic_win_loss), axis=0)
      _input[season].append(train.T)

      _output[season].append(team_to_one_hot[winning_team].T)
      _output[season].append(team_to_one_hot[winning_team].T)

      '''Add game data (twice from both perspectives)'''
      team_data[winning_team].add_data(CSV_GAMES, row)
      team_data[losing_team].add_data(CSV_GAMES, row)



    print("Starting to save examples")
    num_examples = len(_input[season])
    feature_dimension = len(_input[season][0])
    team_one_hot_size = list(team_to_one_hot.values())[0].shape[0]
    for season in sorted(_input.keys()):
      X = np.array(_input[season])
      y = np.array(_output[season])

      X = np.squeeze(X)
      y = np.squeeze(y)

      # Save input and output datasets
      filename = '/home/mark/workspace/SportsAnalytics/MarchMadness/data/season%d.mat' % season
      print("saving %s " % filename)
      scio.savemat(filename, mdict = {'X': X, 'y': y})

    


def load_compact():
  team_to_one_hot = load_teams()
  load_regular_season_games(detailed=False)


if __name__ == "__main__":
  load_regular_season_games()
