import pandas as pd
import numpy as np
import os
import sys
import pickle
import scipy.io as scio
import configs
import hashlib
from configs import AttributeMapper,\
DATA_DIRECTORY, SEASON_FILENAME_COMPACT, SEASON_FILENAME_DETAILED,\
CSV_TEAMS, CSV_SEASON, CSV_REGULAR_SEASON_COMPACT, CSV_REGULAR_SEASON_DETAILED, CSV_SEASON, CSV_TOURNEY_GAMES_COMPACT, CSV_TOURNEY_GAMES_DETAILED, CSV_TOURNEY_SEED, SYMBOL_WIN, SYMBOL_LOSE, CSVSeason


class DataModel:
    def __init__(self, detailed, include_tourney):
      self._detailed = detailed
      self._include_tourney = include_tourney

    @property
    def detailed(self):
      return self._detailed

    @property
    def include_tourney(self):
      return self._include_tourney

    @property
    def description(self):
      if self._detailed and self._include_tourney:
        return 'detailed_with_tourney'
      if self._detailed:
        return 'detailed'
      if self._include_tourney:
        return 'compact_with_tourney'

      return 'compact'

    def __eq__(self, other):
      return self._detailed == other.detailed and self._include_tourney == other.include_tourney



DATA_MODEL_DETAILED = DataModel(True, True)
DATA_MODEL_COMPACT = DataModel(False, True)


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


def load_teams(one_hot=True, force=False):
  if one_hot:
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
  else:
    # Create set of team IDs
    team_ids = set()
    df = pd.read_csv(CSV_TEAMS.get_path())
    for _, row in df.iterrows():
      team_ids.add(int(CSV_TEAMS.get_team_id(row)))
    return team_ids


class SeasonData:
  #MAX_REGULAR_GAMES_PER_SEASON_PER_TEAM = 40   # for non-tournament data
  MAX_REGULAR_GAMES_PER_SEASON_PER_TEAM = 44    # include tournament data
  def __init__(self, season, team_id, data_model):
    '''
      season: year
      team_id: integer
      data_model: used to determine the AttributeMapper to determine what to keep track of.
    '''
    self._season = season
    self._team_id = team_id
    self._data_model = data_model
    self._attributes = {}
    for attribute in AttributeMapper.get_attributes(data_model.detailed):
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
    '''Returns the data for the specified attribute or raises KeyError if attribute DNE'''
    return self._attributes[attribute]


class TeamData:
  def __init__(self, teamid, team_one_hot_encoding, data_model):
    self._team_id = teamid
    self._team_one_hot = team_one_hot_encoding
    self._seasons = dict()
    self._data_model = data_model


  def add_data(self, csvinfo, data_frame):
    '''Add data ..which attributes?''' 
    # Create Training data for this game
    is_winner = csvinfo.get_winning_team(data_frame) == self._team_id
    assert(is_winner or csvinfo.get_losing_team(data_frame) == self._team_id)

    season = csvinfo.get_season(data_frame)
    season_data = self.get_or_create_season_data(season)

    attribute_mapper = AttributeMapper.get_map(self._data_model.detailed, is_winner)
    season_data.add_data(csvinfo, data_frame, attribute_mapper)

  @property
  def team_one_hot(self):
    return self._team_one_hot

  def get_season_win_list(self, csvinfo, data_frame):
    season = csvinfo.get_season(data_frame)
    season_data = self.get_or_create_season_data(season)

    return season_data.get_wins()

  def get_or_create_season_data(self, season):
    if season not in self._seasons:
      self._seasons[season] = SeasonData(season, self._team_id, self._data_model)
    return self._seasons[season]

  def snapshot(self, season, include_hot_streak):
    '''Create input example as a column vector'''
    season_data = self.get_or_create_season_data(season)

    if include_hot_streak:
      season_wins = season_data.wins  # hot streak
    else:
      season_wins = np.empty(0)

    num_games_played = np.count_nonzero(season_data.wins)

    if num_games_played == 0:
      num_games_played = 1

    attribute_list = []
    attributes = sorted(AttributeMapper.get_map(self._data_model.detailed, True).keys())
    for attribute in attributes:
      attribute_list.append(np.sum(season_data.get_attribute(attribute)) / num_games_played)

    attribute_column = np.array(attribute_list).reshape(len(attribute_list), 1)
    snap = np.concatenate((season_wins.reshape(-1,1), attribute_column), axis=0)

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
    

def split_data_to_files(_input, _output, num_splits):
  filenames = []
  _input = np.array_split(_input, num_splits)
  _output = np.array_split(_output, num_splits)

  for split in range(0, num_splits):
    X = np.array(_input[split])
    y = np.array(_output[split])

    X = np.squeeze(X)
    y = np.squeeze(y)
    if (len(y.shape) == 1):
      # Output label is one-dimensional but need to keep format numExamples x numDimensions
      y = y.reshape(-1, 1)
      assert(X.shape[0] == y.shape[0])

    # Save input and output datasets
    filename = '%s.mat' % (configs.all_data_filename(split+1, num_splits))
    print("saving %s with shapes X,y %s, %s" % (filename, X.shape, y.shape))
    filenames.append(filename)
    scio.savemat(filename, mdict = {'X': X, 'y': y})

  return filenames


def load_games(model, data_model, num_historic_win_loss = 10, seasons=range(0, 2020), aggregate_all_data = True, normalize=True, save=False, num_splits=-1):
  '''
  2020 is arbitrary so that it will include all seasons to date
  '''

  if save:
    assert(num_splits >= 1, "Cannot save data into %d number of files" % num_splits)
 
  data_X, data_y, _ = generate_games(model, data_model, seasons, num_historic_win_loss, normalize)

  return parse_seasonal_data(data_X, data_y, seasons=seasons, aggregate_all_data=aggregate_all_data, normalize=normalize, save=save, num_splits=num_splits)


def parse_seasonal_data(data_X, data_y, seasons=range(0, 2020), aggregate_all_data=True, normalize=True, save=False, num_splits=-1):

  print("in seasonal")
  if aggregate_all_data:
    _input = None
    _output = None
    for season in seasons:
      if season in data_X:
        if _input is None:
          _input = data_X[season]
          _output = data_y[season]
        else:
          _input = np.concatenate((_input, data_X[season]), axis=0)
          _output = np.concatenate((_output, data_y[season]), axis=0)

        # Free up memory
        del data_X[season]
        del data_y[season]

    _input = np.squeeze(np.array(_input))
    _output = np.squeeze(np.array(_output))

    if normalize:
      # Normalize all data together
      _input = _input.astype(np.float32)
      mu = np.tile(np.mean(_input, axis=0), (_input.shape[0], 1))
      _input = (_input - mu) / (np.max(_input, axis=0) - np.min(_input, axis=0))
      _input = np.nan_to_num(_input)  # Convert NaNs to 0

    if save:
      return split_data_to_files(_input, _output, num_splits)
    else:
      # Return aggregated data
      return _input, _output
   
  else:
    # Do not aggregate data
    if normalize:
      print("TODO: NOT IMPLEMENTED, normalize over season-split data")

    if save:
      # Save data by seasons
      print("Starting to save examples by season")
      filenames = []
      for season in sorted(_input.keys()):
        X = np.array(_input[season])
        y = np.array(_output[season])

        X = np.squeeze(X)
        y = np.squeeze(y)

        # Save input and output datasets
        filename = '%s.mat' % (configs.season_data_filename(season))
        print("saving %s " % filename)
        filenames.append(filename)
        scio.savemat(filename, mdict = {'X': X, 'y': y})
      return filenames
    else: 
      return data_X, data_y
    
    
  
def generate_games(model, data_model, seasons, NUM_HISTORIC_WIN_LOSS, normalize):
  # TODO: check to see if cached files already exist...would need to update though to determine whether data is detailed or compact and the 'num_historic_win_loss' value used to create data

  if data_model.detailed:
    CSV_REG_GAMES = CSV_REGULAR_SEASON_DETAILED
    CSV_TOU_GAMES = CSV_TOURNEY_GAMES_DETAILED
    SEASON_FILENAME = SEASON_FILENAME_DETAILED
  else:
    print("using compact")
    CSV_REG_GAMES = CSV_REGULAR_SEASON_COMPACT
    CSV_TOU_GAMES = CSV_TOURNEY_GAMES_COMPACT
    SEASON_FILENAME = SEASON_FILENAME_COMPACT

  # dictionaries of season->input examples
  _input = dict()
  _output = dict()

  df = None
  with open(CSV_REG_GAMES.get_path(), 'r') as games_file:
    df = pd.read_csv(games_file)
    if data_model.include_tourney:
      # Merge regular season and tournament games together
      with open(CSV_TOU_GAMES.get_path(), 'r') as tourney_file:
        df = df.merge(pd.read_csv(tourney_file), how='outer')

  print("data size = %d" %(len(df.index)))
  # Keep specified seasons only
  df = df[df[CSV_REG_GAMES.SEASON] >= seasons[0]]
  df = df[df[CSV_REG_GAMES.SEASON] <= seasons[-1]]
  print("new data size = %d" %(len(df.index)))

  # ensure data is chronologically sorted
  df = df.sort_values([CSV_REG_GAMES.SEASON, CSV_REG_GAMES.DAYNUM], ascending=[True, True])
  iteration_count = 0
  
  return generate_examples_from_data(df, model, data_model, NUM_HISTORIC_WIN_LOSS)


  
def get_rival_streak(team1, team2, num_previous_games, df_past_games, CSV):
  '''
    Find the num_previous_games most recent games team1 and team2 have played against each other
    and return a vector of the win-loss wrt team1
  '''
  # limit data to games of team1 vs team2
  team1_as_winner = (df_past_games[CSV.WINNING_TEAM] == team1) & (df_past_games[CSV.LOSING_TEAM] == team2)
  team2_as_winner = (df_past_games[CSV.WINNING_TEAM] == team2) & (df_past_games[CSV.LOSING_TEAM] == team1)
  df_past_games = df_past_games[team1_as_winner | team2_as_winner]

  # sort reverse chronologically
  df_past_games = df_past_games.sort_values([CSV.SEASON, CSV.DAYNUM], ascending=[False, False])

  # apply limit
  df_past_games = df_past_games[:num_previous_games]

  # set 1 for win, -1 for loss wrt team1
  assert(SYMBOL_WIN == 1)
  assert(SYMBOL_LOSE == -1)
  rival_streak = df_past_games[CSV.WINNING_TEAM] == team1
  rival_streak = np.array(rival_streak).astype(int)[::-1] # Convert to ints and have most recent games at right of array..so [10th most recent game, 9th, ..., 2nd, 1st most recent game]
  rival_streak = np.lib.pad(rival_streak, (num_previous_games - rival_streak.shape[0], 0), 'constant', constant_values=(0))
  return rival_streak



def generate_examples_from_data(df, model, data_model, NUM_HISTORIC_WIN_LOSS):
  '''
  Generates the training examples from the specified dataframe.
  '''
  team_to_one_hot = load_teams()
  team_data = {}
  data_hashes = set() # Keep track of duplicate example data

  if data_model.detailed:
    CSV_REG_GAMES = CSV_REGULAR_SEASON_DETAILED
    CSV_TOU_GAMES = CSV_TOURNEY_GAMES_DETAILED
    SEASON_FILENAME = SEASON_FILENAME_DETAILED
  else:
    print("using compact")
    CSV_REG_GAMES = CSV_REGULAR_SEASON_COMPACT
    CSV_TOU_GAMES = CSV_TOURNEY_GAMES_COMPACT
    SEASON_FILENAME = SEASON_FILENAME_COMPACT

  _input = {}
  _output = {}
  iteration_count = 0
  for _, row in df.iterrows():
    winning_team = int(CSV_REG_GAMES.get_winning_team(row))
    losing_team = int(CSV_REG_GAMES.get_losing_team(row))

    season = CSV_REG_GAMES.get_season(row)
    
    iteration_count += 1
    if iteration_count % 3000 == 0:
      print("iteration = %d, season=%d" % (iteration_count, season))
      sys.stdout.flush()


    if winning_team not in team_data:
      '''Create initial empty season data'''
      team_data[winning_team] = TeamData(winning_team, team_to_one_hot[winning_team], data_model)

    if losing_team not in team_data:
      '''Create initial empty season data'''
      team_data[losing_team] = TeamData(losing_team, team_to_one_hot[losing_team], data_model)


    if season not in _input:
      _input[season] = list()
      _output[season] = list()

    # Create two training examples for symmetry
    wteam_snapshot = team_data[winning_team].snapshot(season, model.hot_streak)
    lteam_snapshot = team_data[losing_team].snapshot(season, model.hot_streak)


    # Get rival streak if model allows it
    if model.rival_streak:
      # Need to trim the dataframe to include only games that have occurred up to this point
      previous_seasons = df[CSV_REG_GAMES.SEASON] < season
      current_season = df[CSV_REG_GAMES.SEASON] == season
      current_season &= df[CSV_REG_GAMES.DAYNUM] < CSV_REG_GAMES.get_day_number(row)
      rival_streak = get_rival_streak(
        winning_team, 
        losing_team,
        NUM_HISTORIC_WIN_LOSS,
        df[previous_seasons | current_season],
        CSV_REG_GAMES) 
    else:
      rival_streak = np.empty(0)

    assert(SYMBOL_WIN == 1)
    assert(SYMBOL_LOSE == -1)

    data, labels = model.generate_data(wteam_snapshot, lteam_snapshot, wteam_one_hot=team_to_one_hot[winning_team], lteam_one_hot=team_to_one_hot[losing_team], rival_streak=rival_streak)

    #for i in range(len(labels)):
    #  print(str(i) + ": " + str(data[i]))
    #  print(str(i) + ": " + str(labels[i]))

    for i in range(0, len(labels)):
      data_hash = hashlib.sha1(data[i]).hexdigest()

      if data_hash not in data_hashes:
        _input[season].append(data[i])
        _output[season].append(labels[i])
     
        #print(data[i], labels[i])     
        # add to data hash
        data_hashes.add(data_hash)
    

    # Now that training example has been created we can add it to our knowledge base
    # Add game data (twice from both perspectives)
    team_data[winning_team].add_data(CSV_REG_GAMES, row)
    team_data[losing_team].add_data(CSV_REG_GAMES, row)
    
  for season in _input.keys():
    # Convert lists to np.ndarrays
    _input[season] = np.squeeze(np.array(_input[season]))
    _output[season] = np.squeeze(np.array(_output[season]))
    print("%d has shape %s" %(season, _input[season].shape))


  return _input, _output, team_data

    


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


def load_tournament_teams(season):
  CSV = CSV_TOURNEY_SEED

  df = None
  with open(CSV.get_path(), 'r') as games_file:
    df = pd.read_csv(games_file)
  
  df = df[df[CSV.SEASON] == season]

  return list(CSV.get_team(df))

  

def start_daynum_of_tournament(year):
  df = None
  with open(CSV_TOURNEY_GAMES_COMPACT.get_path(), 'r') as games_file:
    df = pd.read_csv(games_file)
  
  df = df[df[CSV_TOURNEY_GAMES_COMPACT.SEASON] == year]
  df = df.sort_values([CSV_TOURNEY_GAMES_COMPACT.DAYNUM], ascending=[True])

  print("length of df = " + str(len(df)))
  daynum = CSV_TOURNEY_GAMES_COMPACT.get_day_number(df.iloc[0])

  # Make sure detailed dataset reflects the same value
  df = None
  with open(CSV_TOURNEY_GAMES_DETAILED.get_path(), 'r') as games_file:
    df = pd.read_csv(games_file)
  
  df = df[df[CSV_TOURNEY_GAMES_DETAILED.SEASON] == year]
  df = df.sort_values([CSV_TOURNEY_GAMES_DETAILED.DAYNUM], ascending=[True])

  assert(daynum == CSV_TOURNEY_GAMES_DETAILED.get_day_number(df.iloc[0]))

  return daynum
 
def randomize_data(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation, :]
  return shuffled_dataset, shuffled_labels

def read_file(filename):
    if DATA_DIRECTORY in filename:
      filepath = filename
    else:
      filepath = '%s%s' % (DATA_DIRECTORY, filename)
    data = scio.loadmat(filepath)
    return data['X'], data['y']
  

if __name__ == "__main__":
  import models
  # the input/output model to use
  model = models.ModelSymmetrical(hot_streak=False, rival_streak=True)

  # the data model to use
  data_model = DATA_MODEL_COMPACT

  load_games(model, data_model, num_historic_win_loss=10, aggregate_all_data=True, normalize=True, seasons=range(0,1986))
  #load_team_to_region_mapping()
