from season_data import SeasonData
from configs import AttributeMapper

import numpy as np

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



class TeamDataStatisticalRatings:
  '''
  Uses ELO, trueskill
  def __init__(self, teamid, team_one_hot_encoding, data_model):
    self._team_id = teamid
    self._team_one_hot = team_one_hot_encoding
    self._data_model = data_model
    self._elo = ELO()
    self._trueskill = TrueSkill()


  def add_data(self, other_team_data, csvinfo, data_frame):
    #Add data ..which attributes?
    # Create Training data for this game
    is_winner = csvinfo.get_winning_team(data_frame) == self._team_id
    assert(is_winner or csvinfo.get_losing_team(data_frame) == self._team_id)

    if is_winner:
      self._elo = 
      self._trueskill = trueskill.rate_1vs1(self._trueskill, other_team_data._trueskill)
    else:
      self._trueskill = trueskill.rate_1vs1(other_team_data._trueskill, self._trueskill)



  @property
  def team_one_hot(self):
    return self._team_one_hot

  def snapshot(self, season, include_hot_streak):
    # Create input example as a column vector
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
  '''
