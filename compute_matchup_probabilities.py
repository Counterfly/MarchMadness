import configs
import load_data
import models as iomodels
#import model1 as pmodel
import model2 as pmodel
import team_stats

from data_sets import DataSetsFiles

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

NUM_FILE_SPLITS = 12


normalized_min = None
normalized_max = None
normalized_range_min = -1
normalized_range_max = 1

def do_normalize(data, mn, mx, rn, rx):
  min_vec = np.tile(mn, (data.shape[0], 1))
  data = (data - min_vec) / (mx - mn)
  data = np.nan_to_num(data)  # Convert NaNs to 0

  # Should consider normalizing to [rn,rx] range
  data = (data * (rx - rn)) + rn

  return data



def train_model(predictive_model, architecture_model, data_model, teamstats, df_train, num_historic_win_loss, save_model_dir, CSV, normalize=False):

  data, labels = load_data.generate_examples_from_data(df_train, architecture_model, data_model, teamstats, num_historic_win_loss)

  data_partition_fractions = [1.0, 0.0, 0.0]  # Train, Valid, Test

  if normalize:
    data = data.astype(np.float32)

    #mu = np.tile(np.mean(data_tr, axis=0), (data_tr.shape[0], 1))
    # data_tr should depend on data_partition_fractions
    data_tr = data[:int(data.shape[0]*data_partition_fractions[0])]

    # This is poor design...TODO fix
    global normalized_min
    global normalized_max
    global normalized_range_min
    global normalized_range_max
    normalized_min = np.min(data_tr, axis=0)
    normalized_max = np.max(data_tr, axis=0)

    data = do_normalize(data, normalized_min, normalized_max, normalized_range_min, normalized_range_max)
    print("Normalized data", data[1000])

  filenames = load_data.split_data_to_files(data, labels, NUM_FILE_SPLITS)

  

  datasets = DataSetsFiles(filenames, data_partition_fractions, load_data.read_file)

  predictive_model.train(datasets, save_model_dir=save_model_dir)



def inference(architecture_model, data_model, teamstats, saved_model_dir, df_historical, num_historic_win_loss, year, team_ids, outfile, CSV, normalize=False):
  '''
  Does the inference for computing probabilities of team-matchups
  '''

  saver = tf.train.Saver()
  with tf.Session() as session:
    meta_graph_location = saved_model_dir + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

    saver.restore(session, saved_model_dir)

    # Load required session vars
    X_ = tf.get_collection('X_')[0]
    y_ = tf.get_collection('y_')[0]
    keep_prob = tf.get_collection('keep_prob')[0]
    softmax_probabilities = tf.get_collection('softmax_probabilities')[0]
    team_ids = sorted(list(team_ids))

    with open(outfile, 'w') as probfile:
      probfile.write("id,pred\n") # Column headers

      for idx_team1 in range(len(team_ids)):
        team1 = team_ids[idx_team1]
        for idx_team2 in range(idx_team1 + 1, len(team_ids)):
          team2 = team_ids[idx_team2]
          # Generate examples for team1 vs team2
          team1_snapshot, team2_snapshot = teamstats.generate_snapshots(team1, team2, df_historical, architecture_model, data_model)

          # Get rival streak if model allows it
          if architecture_model.rival_streak:
            rival_streak = load_data.get_rival_streak(
              team1, 
              team2,
              num_historic_win_loss,
              df_historical,
              CSV)

            if np.count_nonzero(rival_streak) == 0:
              print("%d, %d have no past" % (team1, team2))
          else:
            rival_streak = np.empty(0)
            
          data_, labels_ = architecture_model.generate_data(team1_snapshot, team2_snapshot, rival_streak=rival_streak)
        
          data = np.array(data_).reshape(2,-1)
          labels = np.array(labels_).reshape(2,-1)

          if normalize:
            global normalized_min
            global normalized_max
            global normalized_range_min
            global normalized_range_max
            data = do_normalize(data, normalized_min, normalized_max, normalized_range_min, normalized_range_max)

          # Just want probabilities, don't do training
          probabilities = session.run(softmax_probabilities, feed_dict = { X_: data, y_: labels, keep_prob: 1.0 })

          
          probabilities = np.squeeze(probabilities).reshape(2,-1)
          # These asserts should be true in ideal world
          #assert(probabilities[0][0] == probabilities[1][1])
          #assert(probabilities[0][1] == probabilities[1][0])


          probability = (probabilities[0][0] + probabilities[1][1]) / 2 
          # output probability wrt team1
          probfile.write("%d_%d_%d,%.4f\n" % (year, team1, team2, probability))
        




def main(tournament_years_to_predict):
  
  num_historic_win_loss = 10

  # the input/output model to use
  arch_model = iomodels.ModelSymmetrical(hot_streak=False, rival_streak=True)
  # the data model to use
  data_model = load_data.DATA_MODEL_DETAILED
  teamstats = team_stats.TeamStatsLifetime()
  NORMALIZE = True


  # Data to train on
  df = None
  CSV = data_model.get_csv()
  df = pd.read_csv(CSV.get_path())
  if data_model.include_tourney:
    # Merge regular season and tournament games together
    csv_tourney = configs.get_tournament_games_csv(data_model.detailed)
    df = df.merge(pd.read_csv(csv_tourney.get_path()), how='outer')
 
  print("data size = %d" % df.shape[0])
  for predict_tournament_year in tournament_years_to_predict: 
    team_ids = load_data.load_tournament_teams(predict_tournament_year)

    # Find the Day number the tournament starts so that we only use historical data for training
    tournament_start_daynum = load_data.start_daynum_of_tournament(predict_tournament_year)

    print("daynum = %d" % tournament_start_daynum)

    # Keep games that occurred prior to tournament start date
    past_seasons = df[CSV.SEASON] < predict_tournament_year
    current_season =  df[CSV.SEASON] == predict_tournament_year
    current_season &= df[CSV.DAYNUM] < tournament_start_daynum
    df_past_games = df[past_seasons | current_season]

    print("new data size = %d" % df_past_games.shape[0])

    # ensure data is chronologically sorted
    df_past_games = df_past_games.sort_values([CSV.SEASON, CSV.DAYNUM], ascending=[True, True])

    save_model_dir = './trained/model1.ckpt'
    train_model(pmodel, arch_model, data_model, teamstats, df_past_games, num_historic_win_loss, save_model_dir, CSV, normalize = NORMALIZE)


    outfile = "probabilities{}.out".format(predict_tournament_year)
    inference(arch_model, data_model, teamstats, save_model_dir, df_past_games, num_historic_win_loss, predict_tournament_year, team_ids, outfile, CSV, normalize=NORMALIZE)


if __name__ == "__main__":
  
  if len(sys.argv) == 2:
    main([int(sys.argv[1])])
  else:
    main([2017])
