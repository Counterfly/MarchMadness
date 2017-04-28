# Models of input features and the output dimensions

import numpy as np
import pandas as pd

class Model(object):
  def __init__(self, desc, log_dir, hot_streak, rival_streak):
    self._description = desc
    self._log_dir = log_dir
    self._hot_streak = hot_streak
    self._rival_streak = rival_streak

  @property
  def description(self):
    return self._description

  @property
  def hot_streak(self):
    return self._hot_streak

  @property
  def rival_streak(self):
    return self._rival_streak

  @property
  def log_dir(self):
    return self._log_dir

  def generate_data(self, wteam_snapshot, lteam_snapshot, wteam_one_hot=np.empty(0), lteam_one_hot=np.empty(0), rival_streak=np.empty(0)):
    raise Exception("UnImplemented Exception")

  def __eq__(self, other):
    return \
      self._description == other.description and\
      self._hot_streak == other.hot_streak and\
      self._rival_streak == other.rival_streak

class ModelWinningTeamOneHot(Model):
  def __init__(self, hot_streak=True, rival_streak=True):
    streak_string = ""
    if hot_streak: streak_string += 'h' 
    if rival_streak: streak_string += 'r'
    super(ModelWinningTeamOneHot, self).__init__(
      'Target is the Winning Team ID',
      '/%s-%s/' % ('wteamid', streak_string),
      hot_streak,
      rival_streak)

  def generate_data(self, wteam_snapshot, lteam_snapshot, wteam_one_hot=np.empty(0), lteam_one_hot=np.empty(0), rival_streak=np.empty(0)):
    data = []
    labels = []

    wteam_snapshot = wteam_snapshot.reshape(-1,1)
    lteam_snapshot = lteam_snapshot.reshape(-1,1)
    wteam_one_hot = wteam_one_hot.reshape(-1,1)
    lteam_one_hot = lteam_one_hot.reshape(-1,1)
    rival_streak = rival_streak.reshape(-1,1)

    data.append(np.concatenate((wteam_one_hot, wteam_snapshot, lteam_one_hot, lteam_snapshot, rival_streak)))
    labels.append(wteam_one_hot)

    rival_streak = np.multiply(-1, rival_streak)
    data.append(np.concatenate((lteam_one_hot, lteam_snapshot, wteam_one_hot, wteam_snapshot, rival_streak)))
    labels.append(wteam_one_hot)

    #print("data labels", len(data), len(labels), data[0].shape, labels[0].shape)
    return data, labels


class ModelSymmetrical(Model):
  def __init__(self, hot_streak=True, rival_streak=True):
    streak_string = ""
    if hot_streak: streak_string += 'h' 
    if rival_streak: streak_string += 'r'
    super(ModelSymmetrical, self).__init__(
      'Output is two nodes where the target is the first or second half team (based on input is symmetrical)',
      '/%s-%s/' % ('symmetrical', streak_string),
      hot_streak,
      rival_streak)

  def generate_data(self, wteam_snapshot, lteam_snapshot, wteam_one_hot=np.empty(0), lteam_one_hot=np.empty(0), rival_streak=np.empty(0)):
    data = []
    labels = []

    wteam_one_hot = wteam_one_hot.reshape(-1,1)
    lteam_one_hot = lteam_one_hot.reshape(-1,1)
    wteam_snapshot = wteam_snapshot.reshape(-1,1)
    lteam_snapshot = lteam_snapshot.reshape(-1,1)
    rival_streak = rival_streak.reshape(-1,1)

    data.append(np.concatenate((wteam_one_hot, wteam_snapshot, lteam_one_hot, lteam_snapshot, rival_streak)))
    #data.append(np.concatenate((wteam_snapshot, lteam_snapshot, rival_streak)))
    labels.append(np.array([1.0, 0.0]))

    rival_streak = np.multiply(-1, rival_streak)
    data.append(np.concatenate((lteam_one_hot, lteam_snapshot, wteam_one_hot, wteam_snapshot, rival_streak)))
    #data.append(np.concatenate((lteam_snapshot, wteam_snapshot, rival_streak)))
    labels.append(np.array([0.0, 1.0]))
    return data, labels




class ModelRNN(object):
  def __init__(self, desc, log_dir, sequence_length):
    self._description = desc
    self._log_dir = log_dir
    self._sequence_length = sequence_length

  @property
  def description(self):
    return self._description

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def log_dir(self):
    return self._log_dir



class ModelRNNRecentGames(ModelRNN):
  def __init__(self, seq_length):
    super(ModelRNNRecentGames, self).__init__(
      'Output is two nodes',
      '/%s/' % ('rnn-recent'),
      seq_length
      )

  def generate_data(self, wteam_sequence, lteam_sequence, wteam_one_hot=np.empty(0), lteam_one_hot=np.empty(0), rival_streak=np.empty(0)):
    data = []
    labels = []

    wteam_one_hot = wteam_one_hot.reshape(-1,1)
    lteam_one_hot = lteam_one_hot.reshape(-1,1)
    sequence = wteam_sequence.merge(lteam_sequence, how='outer')

    data = [sequence]
    labels = []
    
    raise Exception("Not Implemented")

