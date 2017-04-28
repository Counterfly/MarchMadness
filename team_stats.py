from team_data import TeamData
from configs import AttributeMapper

import numpy as np


class TeamStatsAbstract(object):

  def __init__(self):
    pass

  def generate_snapshots(self, team1, team2, dataframe, iomodel, data_model):
    # TODO: add rival and hot streaks to be auto added in here instead of in load_data
    # TODO: right now hot streaks aren't even added if set in iomodel (because prior validation said it was essentially meaningless but it should be re-added for completeness
    raise Exception("Not Implemented")

  def determine_most_recent_game(self, team1, team2, dataframe, data_model):
    # Finds the last row (most recent chronologically) with team1 and team2
    CSV = data_model.get_csv()
    team1_as_winner = (dataframe[CSV.WINNING_TEAM] == team1) & (dataframe[CSV.LOSING_TEAM] == team2)
    team2_as_winner = (dataframe[CSV.WINNING_TEAM] == team2) & (dataframe[CSV.LOSING_TEAM] == team1)

    dataframe = dataframe[team1_as_winner | team2_as_winner]
    
    return dataframe.tail(1)


class TeamStatsLifetime(TeamStatsAbstract):

  def __init__(self):
    super(TeamStatsLifetime, self).__init__()


  def generate_snapshots(self, team1, team2, dataframe, iomodel, data_model):
    # Temporal data so make sure data frame is chronological
    #dataframe = dataframe.sort_values([CSV_REG_GAMES.SEASON, CSV_REG_GAMES.DAYNUM], ascending=[True, True])

    CSV = data_model.get_csv()
    row = self.determine_most_recent_game(team1, team2, dataframe, data_model)

    wteam_attributes = dict()
    lteam_attributes = dict()
    for attribute in AttributeMapper.get_attributes(data_model.detailed):
      wteam_attributes[attribute] = 0
      lteam_attributes[attribute] = 0

      num_games = 1
    if row.empty:
      #print("Row Should not be EMPTY!!! teams= %d,%d" % (team1, team2))
      num_games=1 # NOOP
    else:
      winning_team = int(CSV.get_winning_team(row))
      losing_team = int(CSV.get_losing_team(row))
      season = int(CSV.get_season(row))
      daynum = int(CSV.get_day_number(row))

      # This is only thing that differs across teamstats objects
      # Generate snapshots for both teams
      past_season = dataframe[CSV.SEASON] < season
      current_season = dataframe[CSV.SEASON] == season
      current_season &= dataframe[CSV.DAYNUM] < daynum
      df_past_games = dataframe[past_season | current_season]
      
      winning_team_as_winner = (df_past_games[CSV.WINNING_TEAM] == winning_team)
      winning_team_as_loser  = (df_past_games[CSV.LOSING_TEAM] == winning_team)
      losing_team_as_winner  = (df_past_games[CSV.WINNING_TEAM] == losing_team)
      losing_team_as_loser   = (df_past_games[CSV.LOSING_TEAM] == losing_team)

      df_winning_team_as_winner = df_past_games[winning_team_as_winner].sum(axis=0)
      df_winning_team_as_loser = df_past_games[winning_team_as_loser].sum(axis=0)
      df_losing_team_as_winner = df_past_games[losing_team_as_winner].sum(axis=0)
      df_losing_team_as_loser = df_past_games[losing_team_as_loser].sum(axis=0)

      for attribute, df_key in AttributeMapper.get_map(data_model.detailed, True).items():
        wteam_attributes[attribute] += int(df_winning_team_as_winner[df_key])
        lteam_attributes[attribute] += int(df_losing_team_as_winner[df_key])
    

      for attribute, df_key in AttributeMapper.get_map(data_model.detailed, False).items():
        wteam_attributes[attribute] += int(df_winning_team_as_loser[df_key])
        lteam_attributes[attribute] += int(df_losing_team_as_loser[df_key])


      num_games = max(1, np.count_nonzero(winning_team_as_winner) + np.count_nonzero(losing_team_as_winner))

    wteam_snapshot = [wteam_attributes[k]//num_games for k in sorted(wteam_attributes.keys())]
    lteam_snapshot = [lteam_attributes[k]//num_games for k in sorted(lteam_attributes.keys())]
    return np.array(wteam_snapshot), np.array(lteam_snapshot)

class TeamStatsSeasonal(TeamStatsAbstract):

  def __init__(self):
    super(TeamStatsSeasonal, self).__init__()


  def generate_snapshots(self, team1, team2, dataframe, iomodel, data_model):
    # Temporal data so make sure data frame is chronological
    #dataframe = dataframe.sort_values([CSV_REG_GAMES.SEASON, CSV_REG_GAMES.DAYNUM], ascending=[True, True])

    CSV = data_model.get_csv()
    row = self.determine_most_recent_game(team1, team2, dataframe, data_model)

    wteam_attributes = dict()
    lteam_attributes = dict()
    for attribute in AttributeMapper.get_attributes(data_model.detailed):
      wteam_attributes[attribute] = 0
      lteam_attributes[attribute] = 0

      num_games = 1
    if row.empty:
      #print("Row Should not be EMPTY!!! teams= %d,%d" % (team1, team2))
      num_games=1
    else:
      winning_team = int(CSV.get_winning_team(row))
      losing_team = int(CSV.get_losing_team(row))
      season = int(CSV.get_season(row))
      daynum = int(CSV.get_day_number(row))

      # This is only thing that differs across teamstats objects
      # Generate snapshots for both teams
      current_season = dataframe[CSV.SEASON] == season
      current_season &= dataframe[CSV.DAYNUM] < daynum
      df_past_games = dataframe[current_season]
      
      winning_team_as_winner = (df_past_games[CSV.WINNING_TEAM] == winning_team)
      winning_team_as_loser  = (df_past_games[CSV.LOSING_TEAM] == winning_team)
      losing_team_as_winner  = (df_past_games[CSV.WINNING_TEAM] == losing_team)
      losing_team_as_loser   = (df_past_games[CSV.LOSING_TEAM] == losing_team)

      df_winning_team_as_winner = df_past_games[winning_team_as_winner].sum(axis=0)
      df_winning_team_as_loser = df_past_games[winning_team_as_loser].sum(axis=0)
      df_losing_team_as_winner = df_past_games[losing_team_as_winner].sum(axis=0)
      df_losing_team_as_loser = df_past_games[losing_team_as_loser].sum(axis=0)

      for attribute, df_key in AttributeMapper.get_map(data_model.detailed, True).items():
        wteam_attributes[attribute] += int(df_winning_team_as_winner[df_key])
        lteam_attributes[attribute] += int(df_losing_team_as_winner[df_key])
    

      for attribute, df_key in AttributeMapper.get_map(data_model.detailed, False).items():
        wteam_attributes[attribute] += int(df_winning_team_as_loser[df_key])
        lteam_attributes[attribute] += int(df_losing_team_as_loser[df_key])
        
      
      num_games = max(1, np.count_nonzero(winning_team_as_winner) + np.count_nonzero(losing_team_as_winner))

    wteam_snapshot = [wteam_attributes[k]//num_games for k in sorted(wteam_attributes.keys())]
    lteam_snapshot = [lteam_attributes[k]//num_games for k in sorted(lteam_attributes.keys())]
    return np.array(wteam_snapshot), np.array(lteam_snapshot)




class TeamStatsPairMatchups(TeamStatsAbstract):

  def __init__(self):
    super(TeamStatsPairMatchups, self).__init__()


  def generate_snapshots(self, team1, team2, dataframe, iomodel, data_model):
    # Temporal data so make sure data frame is chronological
    #dataframe = dataframe.sort_values([CSV_REG_GAMES.SEASON, CSV_REG_GAMES.DAYNUM], ascending=[True, True])

    CSV = data_model.get_csv()
    row = self.determine_most_recent_game(team1, team2, dataframe, data_model)

    wteam_attributes = dict()
    lteam_attributes = dict()
    for attribute in AttributeMapper.get_attributes(data_model.detailed):
      wteam_attributes[attribute] = 0
      lteam_attributes[attribute] = 0

      num_games = 1
    if row.empty:
      #print("Row Should not be EMPTY!!! teams= %d,%d" % (team1, team2))
      num_games=1
    else:
      winning_team = int(CSV.get_winning_team(row))
      losing_team = int(CSV.get_losing_team(row))
      season = int(CSV.get_season(row))
      daynum = int(CSV.get_day_number(row))

      # Generate snapshots for both teams
      previous_seasons = dataframe[CSV.SEASON] < season
      current_season = dataframe[CSV.SEASON] == season
      current_season &= dataframe[CSV.DAYNUM] < daynum
      df_past_games = dataframe[previous_seasons | current_season]
     
      # These mean: "current winner that also won in previous games" 
      winning_team_as_winner = (df_past_games[CSV.WINNING_TEAM] == winning_team) & (df_past_games[CSV.LOSING_TEAM] == losing_team)
      losing_team_as_winner = (df_past_games[CSV.WINNING_TEAM] == losing_team) & (df_past_games[CSV.LOSING_TEAM] == winning_team)

      df_winning_team_as_winner = df_past_games[winning_team_as_winner].sum(axis=0)
      df_losing_team_as_winner = df_past_games[losing_team_as_winner].sum(axis=0)
      for attribute, df_key in AttributeMapper.get_map(data_model.detailed, True).items():
        wteam_attributes[attribute] += int(df_winning_team_as_winner[df_key])
        lteam_attributes[attribute] += int(df_losing_team_as_winner[df_key])
    

      for attribute, df_key in AttributeMapper.get_map(data_model.detailed, False).items():
        wteam_attributes[attribute] += int(df_losing_team_as_winner[df_key])
        lteam_attributes[attribute] += int(df_winning_team_as_winner[df_key])
        
      
      num_games = max(1, np.count_nonzero(winning_team_as_winner) + np.count_nonzero(losing_team_as_winner))

    wteam_snapshot = [int(wteam_attributes[k])//num_games for k in sorted(wteam_attributes.keys())]
    lteam_snapshot = [int(lteam_attributes[k])//num_games for k in sorted(lteam_attributes.keys())]

    return np.array(wteam_snapshot), np.array(lteam_snapshot)



class TeamStatsNull(TeamStatsAbstract):

  def __init__(self):
    pass

  def generate_snapshots(self, team1, team2, dataframe, iomodel, data_model):
    return np.empty(0), np.empty(0)






class TeamStatsRecentGameSequence(TeamStatsAbstract):
  # Outputs the past X Games as a sequence
  # Used for RNN

  def __init__(self, num_games):
    super(TeamStatsRecentGameSequence, self).__init__()
    self._num_games = num_games


  def generate_snapshots(self, team1, team2, dataframe, iomodel, data_model):
    # Temporal data so make sure data frame is chronological
    #dataframe = dataframe.sort_values([CSV_REG_GAMES.SEASON, CSV_REG_GAMES.DAYNUM], ascending=[True, True])

    CSV = data_model.get_csv()
    row = self.determine_most_recent_game(team1, team2, dataframe, data_model)

    sequences = np.zeros((self._num_games, row.shape[1]))
    if row.empty:
      #print("Row Should not be EMPTY!!! teams= %d,%d" % (team1, team2))
      num_games=1
    else:
      winning_team = int(CSV.get_winning_team(row))
      losing_team = int(CSV.get_losing_team(row))
      season = int(CSV.get_season(row))
      daynum = int(CSV.get_day_number(row))

      # Generate snapshots for both teams
      previous_seasons = dataframe[CSV.SEASON] < season
      current_season = dataframe[CSV.SEASON] == season
      current_season &= dataframe[CSV.DAYNUM] < daynum
      df_past_games = dataframe[previous_seasons | current_season]
    
      # Get all past games for each team
      winning_team_games = (df_past_games[CSV.WINNING_TEAM] == winning_team) | (df_past_games[CSV.LOSING_TEAM] == winning_team)
      losing_team_games = (df_past_games[CSV.WINNING_TEAM] == losing_team) | (df_past_games[CSV.LOSING_TEAM] == losing_team)

      df_winning_team_games = df_past_games[winning_team_games]._get_numeric_data()
      df_losing_team_games  = df_past_games[losing_team_games]._get_numeric_data()

      # Get most recent games
      df_winning_team_games = df_winning_team_games.sort_values([CSV.SEASON, CSV.DAYNUM], ascending=[False, False])
      df_losing_team_games = df_losing_team_games.sort_values([CSV.SEASON, CSV.DAYNUM], ascending=[False, False])


      #df_merged = df_winning_team_games.merge(df_losing_team_games, how='outer')
      #df_merged = df_merged.sort_values([CSV.SEASON, CSV.DAYNUM], ascending=[False, False])[:self._num_games, :]

      #sequences = pandas.as_matrix(columns=None).T  # Could customize the columns but need both teamIDs
      
      return df_winning_team_games, df_losing_team_games
