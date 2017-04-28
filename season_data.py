from configs import AttributeMapper, SYMBOL_LOSE, SYMBOL_WIN

import numpy as np

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
