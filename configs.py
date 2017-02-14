import pandas as pd

class CSVInfo(object):
  def __init__(self, filepath):
    self.file_path = filepath

  def get_path(self):
    return self.file_path




class CSVTeam(CSVInfo):
  TEAM_ID = 'Team_Id'
  TEAM_NAME = 'Team_Name'

  def __init__(self, filepath):
    super(CSVTeam, self).__init__(filepath)

  def get_team_id(self, csv_row):
    return str(csv_row[self.TEAM_ID])
    

class CSVRegularSeasonCompact(CSVInfo):
  DAYNUM = 'Daynum'
  SEASON = 'Season'
  WINNING_SCORE = 'Wscore'
  WINNING_TEAM = 'Wteam'
  LOSING_SCORE = 'Lscore'
  LOSING_TEAM = 'Lteam'
  WINNING_LOCATION = 'Wloc'
  NUM_OVERTIMES = 'Numot'

  def __init__(self, filepath):
    super(CSVRegularSeasonCompact, self).__init__(filepath)

  #def get_attribute(self, csv_row, attr):
  #  return csv_row[attr]

  def get_season(self, csv_row):
    return csv_row[self.SEASON]

  def get_day_number(self, csv_row):
    return csv_row[self.DAYNUM]

  def get_winning_team(self, csv_row):
    return csv_row[self.WINNING_TEAM]

  def get_winning_score(self, csv_row):
    return csv_row[self.WINNING_SCORE]

  def get_losing_team(self, csv_row):
    return csv_row[self.LOSING_TEAM]

  def get_losing_score(self, csv_row):
    return csv_row[self.LOSING_SCORE]



class CSVRegularSeasonDetailed(CSVRegularSeasonCompact):
  WINNING_ASSISTS = 'Wast'
  WINNING_BLOCKS = 'Wblk'
  WINNING_FIELD_GOALS_ATTEMPTED = 'Wfga'
  WINNING_FIELD_GOALS_MADE = 'Wfgm'
  WINNING_FREE_THROWS_ATTEMPTED = 'Wfta'
  WINNING_FREE_THROWS_MADE = 'Wftm'
  WINNING_PERSONAL_FOULS = 'Wpf'
  WINNING_REBOUNDS_DEFENSIVE = 'Wdr'
  WINNING_REBOUNDS_OFFENSIVE = 'Wor'
  WINNING_STEALS = 'Wstl'
  WINNING_THREE_POINTS_ATTEMPTED = 'Wfga3'
  WINNING_THREE_POINTS_MADE = 'Wfgm3'
  WINNING_TURNOVERS = 'Wto'
  LOSING_ASSISTS = 'Last'
  LOSING_BLOCKS = 'Lblk'
  LOSING_FIELD_GOALS_ATTEMPTED = 'Lfga'
  LOSING_FIELD_GOALS_MADE = 'Lfgm'
  LOSING_FREE_THROWS_ATTEMPTED = 'Lfta'
  LOSING_FREE_THROWS_MADE = 'Lftm'
  LOSING_PERSONAL_FOULS = 'Lpf'
  LOSING_REBOUNDS_DEFENSIVE = 'Ldr'
  LOSING_REBOUNDS_OFFENSIVE = 'Lor'
  LOSING_STEALS = 'Lstl'
  LOSING_THREE_POINTS_ATTEMPTED = 'Lfga3'
  LOSING_THREE_POINTS_MADE = 'Lfgm3'
  LOSING_TURNOVERS = 'Lto'

  def __init__(self, filepath):
    super(CSVRegularSeasonDetailed, self).__init__(filepath)


class AttributeMapper:
  ATTRIBUTE_MAP_WINNING_TEAM = {
    'against': CSVRegularSeasonDetailed.LOSING_TEAM,
    'assists': CSVRegularSeasonDetailed.WINNING_ASSISTS,
    'blocks': CSVRegularSeasonDetailed.WINNING_BLOCKS,
    'fieldgoalsattempted': CSVRegularSeasonDetailed.WINNING_FIELD_GOALS_ATTEMPTED,
    'fieldgoalsmade': CSVRegularSeasonDetailed.WINNING_FIELD_GOALS_MADE,
    'freethrowsattempted': CSVRegularSeasonDetailed.WINNING_FREE_THROWS_ATTEMPTED,
    'freethrowsmade': CSVRegularSeasonDetailed.WINNING_FREE_THROWS_MADE,
    'personalfouls': CSVRegularSeasonDetailed.WINNING_PERSONAL_FOULS,
    'reboundsdefensive': CSVRegularSeasonDetailed.WINNING_REBOUNDS_DEFENSIVE,
    'reboundsoffensive': CSVRegularSeasonDetailed.WINNING_REBOUNDS_OFFENSIVE,
    'score': CSVRegularSeasonDetailed.WINNING_SCORE,
    'scoreagainst': CSVRegularSeasonDetailed.LOSING_SCORE,
    'steals': CSVRegularSeasonDetailed.WINNING_STEALS,
    'threepointsattempted': CSVRegularSeasonDetailed.WINNING_THREE_POINTS_ATTEMPTED,
    'threepointsmade': CSVRegularSeasonDetailed.WINNING_THREE_POINTS_MADE,
    'turnovers': CSVRegularSeasonDetailed.WINNING_TURNOVERS
  }

  ATTRIBUTE_MAP_LOSING_TEAM = {
    'against': CSVRegularSeasonDetailed.WINNING_TEAM,
    'assists': CSVRegularSeasonDetailed.LOSING_ASSISTS,
    'blocks': CSVRegularSeasonDetailed.LOSING_BLOCKS,
    'fieldgoalsattempted': CSVRegularSeasonDetailed.LOSING_FIELD_GOALS_ATTEMPTED,
    'fieldgoalsmade': CSVRegularSeasonDetailed.LOSING_FIELD_GOALS_MADE,
    'freethrowsattempted': CSVRegularSeasonDetailed.LOSING_FREE_THROWS_ATTEMPTED,
    'freethrowsmade': CSVRegularSeasonDetailed.LOSING_FREE_THROWS_MADE,
    'personalfouls': CSVRegularSeasonDetailed.LOSING_PERSONAL_FOULS,
    'reboundsdefensive': CSVRegularSeasonDetailed.LOSING_REBOUNDS_DEFENSIVE,
    'reboundsoffensive': CSVRegularSeasonDetailed.LOSING_REBOUNDS_OFFENSIVE,
    'score': CSVRegularSeasonDetailed.LOSING_SCORE,
    'scoreagainst': CSVRegularSeasonDetailed.WINNING_SCORE,
    'steals': CSVRegularSeasonDetailed.LOSING_STEALS,
    'threepointsattempted': CSVRegularSeasonDetailed.LOSING_THREE_POINTS_ATTEMPTED,
    'threepointsmade': CSVRegularSeasonDetailed.LOSING_THREE_POINTS_MADE,
    'turnovers': CSVRegularSeasonDetailed.LOSING_TURNOVERS
  }
  assert(set(ATTRIBUTE_MAP_WINNING_TEAM.keys()) == set(ATTRIBUTE_MAP_LOSING_TEAM.keys()))

  def __init__(self):
    pass
  
  @classmethod
  def get_map(cls, is_winner):
    if is_winner:
      return cls.ATTRIBUTE_MAP_WINNING_TEAM
    
    return cls.ATTRIBUTE_MAP_LOSING_TEAM

  @classmethod
  def get_attributes(cls):
    return cls.ATTRIBUTE_MAP_WINNING_TEAM.keys()


class CSVTourneySeed(CSVInfo):
  SEASON = 'Season'
  SEED = 'Seed'
  TEAM = 'Team'
  
  def __init__(self, filepath):
    super(CSVTourneySeed, self).__init__(filepath)

  def get_season(self, csv_row):
    return csv_row[self.SEASON]

  def get_team(self, csv_row):
    return csv_row[self.TEAM]

  def get_seed(self, csv_row):
    return csv_row[self.SEED]

  def get_region_code(self, csv_row):
    # Region code is always the first letter in the seed (W X Y or Z)
    return self.get_seed(csv_row)[0]

  def get_region_seed(self, csv_row):
    # gets the seed for the region
    return self.get_seed(csv_row)[1:]
  
class CSVSeason(CSVInfo):
  SEASON = 'Season'
  REGION_W = 'Regionw'
  REGION_X = 'Regionx'
  REGION_Y = 'Regiony'
  REGION_Z = 'Regionz'
  
  def __init__(self, filepath):
    super(CSVSeason, self).__init__(filepath)

  def get_season(self, csv_row):
    return csv_row[self.SEASON]

  def get_region_attribute_from_code(self, region_code):
    if region_code == 'W' or region_code == 'w':
      return self.REGION_W
    elif region_code == 'X' or region_code == 'x':
      return self.REGION_X
    elif region_code == 'Y' or region_code == 'y':
      return self.REGION_Y
    elif region_code == 'Z' or region_code == 'z':
      return self.REGION_Z
    else:
      raise KeyError('no region code %s' % region_code)

 # def load_data(self):
 #   self._data = pd.read_csv(self.get_path())

 # def get_attribute(self, attribute):
 #   self._data[]
    

DATA_DIRECTORY = "/home/mark/workspace/SportsAnalytics/MarchMadness/data/"
SEASON_FILENAME_COMPACT = 'compact_season'
SEASON_FILENAME_DETAILED = 'detailed_season'

CSV_TEAMS = CSVTeam(DATA_DIRECTORY + 'Teams.csv')

# Regular Season stats
CSV_REGULAR_SEASON_COMPACT  = CSVRegularSeasonCompact(DATA_DIRECTORY + 'RegularSeasonCompactResults.csv')
CSV_REGULAR_SEASON_DETAILED = CSVRegularSeasonCompact(DATA_DIRECTORY + 'RegularSeasonDetailedResults.csv')

# Tournament Seed
CSV_TOURNEY_SEED = CSVTourneySeed(DATA_DIRECTORY + 'TourneySeeds.csv')

# Season
CSV_SEASON = CSVSeason(DATA_DIRECTORY + 'Seasons.csv')

# Magic Numbers
SYMBOL_WIN = 1
SYMBOL_LOSE = -1
# with 0 meaning game not played
