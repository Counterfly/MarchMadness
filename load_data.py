import pandas as pd
import numpy as np
import os
import sys
import pickle
import scipy.io as scio
from configs import AttributeMapper,\
DATA_DIRECTORY, SEASON_FILENAME_COMPACT, SEASON_FILENAME_DETAILED,\
CSV_TEAMS, CSV_SEASON, CSV_REGULAR_SEASON_COMPACT, CSV_REGULAR_SEASON_DETAILED, CSV_SEASON, CSV_TOURNEY_SEED, SYMBOL_WIN, SYMBOL_LOSE, CSVSeason


np.set_printoptions(suppress=True)

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
    for _, row in df.iterrows():
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

    if num_games_played == 0:
      num_games_played = 1

    attribute_list = []
    attributes = sorted(AttributeMapper.get_map(True).keys())
    for attribute in attributes:
      attribute_list.append(np.sum(season_data.get_attribute(attribute)) / num_games_played)

    attribute_column = np.array(attribute_list).reshape(len(attribute_list), 1)
    snap = np.concatenate((self.team_one_hot.reshape(-1,1), season_wins.reshape(-1,1), attribute_column), axis=0)

    return snap



class HistoricGames:
  def __init__(self, horizon):
    '''
    horizon is the number of games to keep between matchups
    '''
    self._history = {}
    self._horizon = horizon

  def get_team_tuple(self, team1, team2):
    '''
    Store games with key as the lesser team (team1 <= team2 => team1 is key)
    '''
    if team1 <= team2:
      return (team1, team2)
    
    else:
       return (team2, team1)

  def add_game(self, team1, team2, winning_team):
    '''
    winning array is stored like [winner of horizon^th most recent game, ..., winner of most recent game]
    '''
    team_tuple = self.get_team_tuple(team1, team2)

    if team_tuple[0] not in self._history:
      self._history[team_tuple[0]] = {}
    
    if team_tuple[1] not in self._history[team_tuple[0]]:
      self._history[team_tuple[0]][team_tuple[1]] = np.zeros(self._horizon)

    winning_teams = self._history[team_tuple[0]][team_tuple[1]]
    # get last game number
    game_number = 0
    while game_number < self._horizon and winning_teams[game_number] != 0:
      game_number += 1
   
    if game_number < self._horizon:
      # can add new game without removing any previous
      winning_teams[game_number] = winning_team 
    else:
      # Need to shift values and store only horizon most recent games
      for i in range(1,len(winning_teams)):
        winning_teams[i-1] = winning_teams[i]
      winning_teams[self._horizon - 1] = winning_team 

  def get_historic_win_loss(self, team1, team2):
    '''
    Use SYMBOL_WIN and SYMBOL_LOSE as identifiers for whether team won or lost
    win loss is wrt team1
    '''
    team_tuple = self.get_team_tuple(team1, team2)

    assert(team_tuple[0] in self._history)
    assert(team_tuple[1] in self._history[team_tuple[0]])

    winning_team = self._history[team_tuple[0]][team_tuple[1]].copy()

    winning_team[(winning_team == team1)] = SYMBOL_WIN
    winning_team[(winning_team == team2)] = SYMBOL_LOSE
    return winning_team
    
    
def load_regular_season_games(detailed=True, num_historic_win_loss = 10, seasons=range(2003, 2017)):
 
  generate_regular_season_games(detailed, num_historic_win_loss)

  if detailed:
    SEASON_FILENAME = SEASON_FILENAME_DETAILED
  else:
    SEASON_FILENAME = SEASON_FILENAME_COMPACT

  _input = list()
  _output = list()
  for season in seasons:
    filename = '%s%s%d.mat' % (DATA_DIRECTORY, SEASON_FILENAME, season)
    data = scio.loadmat(filename) # may fail t load based on seasons, fine, should exit program
    _input.append(data['X'])
    _output.append(data['y'])

  return np.array(_input), np.array(_output)
    
    
  
def generate_regular_season_games(detailed=True, NUM_HISTORIC_WIN_LOSS=10):
  def get_historic_win_loss(team1, team2, num_previous_games):
    '''
      Find the num_previous_games most recent games team1 and team2 have played against each other
      and return a vector of the win-loss wrt team1
    '''
    return np.zeros((num_previous_games, 1))

  if detailed:
    CSV_GAMES = CSV_REGULAR_SEASON_DETAILED
    SEASON_FILENAME = SEASON_FILENAME_DETAILED
  else:
    CSV_GAMES = CSV_REGULAR_SEASON_COMPACT
    SEASON_FILENAME = SEASON_FILENAME_COMPACT


  team_to_one_hot = load_teams()
  team_data = {}
  historic_games = HistoricGames(10)

  LAST_SEASON=2017

  # TODO: check to see if cached files already exist...would need to update though to determine whether data is detailed or compact and the 'num_historic_win_loss' value used to create data

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

      # Now that training example has been created we can add it to our knowledge base
      
      #Add game data (twice from both perspectives)
      team_data[winning_team].add_data(CSV_GAMES, row)
      team_data[losing_team].add_data(CSV_GAMES, row)
      
      # Add to historic games
      historic_games.add_game(winning_team, losing_team, winning_team)


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
      filename = '%s%s%d.mat' % (DATA_DIRECTORY, SEASON_FILENAME, season)
      print("saving %s " % filename)
      scio.savemat(filename, mdict = {'X': X, 'y': y})


    return _input, _output

    


def load_compact():
  team_to_one_hot = load_teams()
  load_regular_season_games(detailed=False)



def load_seasons():
  # loads the Season data frame
  # this should have a single row per season with the regions associated with each season
  # season, regionW, regionX, regionY, regionZ, anything else as well
  return pd.read_csv(CSV_SEASON.get_path())


def load_team_to_region_mapping(force = False):
  '''
  This takes creates a dictionary that maps teamid to the region of interest that they were placed in for their last tournament.
  DEPENDENCIES:
    'Season.csv': season, date, and region list in W,X,Y,Z format
    'TourneySeeds.csv': season, seed, team
  '''

  team_to_region_file = DATA_DIRECTORY + 'teams_to_region.pickle'

  if force or not os.path.exists(team_to_region_file):
    REGIONS = set(['South', 'East', 'West', 'Midwest']) # Regions of interest
    num_teams = len(load_teams())
    team_to_region = {}
    season_to_region_df = load_seasons()

    region_count = dict.fromkeys(REGIONS, 0)
    # Iterate rows in dataframe keeping the most recent team region
    #  reverse order with early termination would be nice but since
    #  every team is not guaranteed to have played in a tournament then
    #  it will just add extra computational burden..except now with region_count
    #  both methods are probably the same
    for _, row in pd.read_csv(CSV_TOURNEY_SEED.get_path()).iterrows():
      season = CSV_TOURNEY_SEED.get_season(row)
      region_code = CSV_TOURNEY_SEED.get_region_code(row)

      region_attribute = CSV_SEASON.get_region_attribute_from_code(region_code)
      region = season_to_region_df[(season_to_region_df[CSVSeason.SEASON] == season)][region_attribute].get_values()[0]

      if region in REGIONS:
        team = CSV_TOURNEY_SEED.get_team(row)
        
        if team in team_to_region: 
          region_count[team_to_region[team]] -= 1
      
        region_count[region] += 1

        # Only add it if it is a region of interest
        team_to_region[team] = region
 
        if len(team_to_region) == num_teams:
          break

    # Make sure don't have more teams in tournaments than actual teams
    assert(len(team_to_region) <= num_teams)
    print("%d teams are not in tournaments" % (num_teams - len(team_to_region)))

    print(region_count)
    # Store mapping in file      
    with open(team_to_region_file, 'wb') as out_file:
      pickle.dump(team_to_region, out_file, protocol=pickle.HIGHEST_PROTOCOL)

    return team_to_region
  else:
    print('%s already exists' % team_to_region_file)

    # Load data and read in as numpy array
    with open(team_to_region_file, 'rb') as team_file:
      team_to_region = pickle.load(team_file)

    return team_to_region


  
  

if __name__ == "__main__":
  load_regular_season_games()
  #load_team_to_region_mapping()
