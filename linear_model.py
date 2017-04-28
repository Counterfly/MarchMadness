import math
import numpy as np
import tensorflow as tf
import os

# User defined modules
import load_data
import models
import team_stats
from data_sets import DataSetsFiles
from flag_values import _FlagValues as FlagValues


def train(datasets, log_dir=None, log_description=None, save_model_dir=None):
  FLAGS = FlagValues()
  FLAGS.learning_rate = 1e-4
  FLAGS.summaries_dir = './logs'

  num_input_features = datasets.num_features
  num_classification_labels = datasets.num_classification_labels # number nodes in output layer


  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
  session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

  def weight_variable(shape, stddev=0.1, name=''):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

  def bias_variable(shape, const=0.1, name=''):
    initial = tf.constant(const, shape=shape)
    return tf.Variable(initial, name=name)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Taken from tensorflow's tutorial.
    """
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, weights, biases, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu (as default) to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.

    Adapted from tensorflow's tutorial.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights)
      with tf.name_scope('biases'):
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  layer1 = {
    'in': num_input_features,
    'out': num_classification_labels
  }

  X_ = tf.placeholder(tf.float32, shape=(None, layer1['in']), name='X_')
  tf.add_to_collection('X_', X_)
  y_ = tf.placeholder(tf.float32, shape=(None, num_classification_labels), name='y_')
  tf.add_to_collection('y_', y_)



  # Model
  # Variables.
  weights_layer1 = weight_variable([layer1['in'], layer1['out']], name='w1')
  biases_layer1 = bias_variable([layer1['out']], name='b1')
  tf.add_to_collection('vars', weights_layer1)
  tf.add_to_collection('vars', biases_layer1)

  # Do not apply softmax yet
  output_activation = nn_layer(X_, weights_layer1, biases_layer1, 'output_activation', tf.identity)

  # This is only used to read output probability, doesn't serve in remainder computations
  softmax_probabilities = tf.nn.softmax(output_activation)
  tf.add_to_collection('softmax_probabilities', softmax_probabilities)

  # Training computation.
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_activation, labels=y_))

  beta = 0.0001
  loss = loss + \
        beta * tf.nn.l2_loss(weights_layer1)
  tf.add_to_collection('loss', loss)
      
  # TensorBoard
  tf.summary.scalar('cross_entropy', loss)

  # Optimizer.
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

  with tf.name_scope('accuracy'): 
    # Predictions for the training, validation, and test data.
    # predictions don't need softmax if simply choosing max. softmax won't alter ranking
    predictions = tf.argmax(output_activation, 1)
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(predictions, tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #accuracy = tf.scalar_mul(100, tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
  tf.summary.scalar('accuracy', accuracy)
  tf.add_to_collection('accuracy', accuracy)

  # Merge summaries
  merged = tf.summary.merge_all()
  if log_dir:
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '%s/%s/%s' % (log_dir, log_description, 'train'), session.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '%s/%s/%s' % (log_dir, log_description, 'valid'))
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '%s/%s/%s' % (log_dir, log_description, 'test'))
    

                                   
  #####
  batch_size = 128
  num_steps = 10000
  print("Num steps = %d" % num_steps)
    
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(1, num_steps):
    # Generate minibatch.
    batch_data, batch_labels = datasets.train.next_batch(batch_size)

    feed_dict = {X_ : batch_data, y_ : batch_labels }
    summary, loss_, _, acc = session.run([merged, loss, optimizer, accuracy], feed_dict=feed_dict)

    if log_dir:
      train_writer.add_summary(summary, step)

    if (step % (num_steps//50) == 0):
      print("\n===\nstep: %d" % step)
      print("Minibatch loss: %f" % loss_)
      print("Minibatch accuracy: %.5f" % acc) #accuracy.eval(feed_dict={X_: batch_data, y_: batch_labels }))
  
      if datasets.valid:
        summary_valid, acc_loss, acc_valid = session.run([merged, loss, accuracy], feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels })
        print("Validation loss: %f" % acc_loss)
        print("Validation accuracy: %.5f" % acc_valid)
        if log_dir:
          valid_writer.add_summary(summary_valid, step)

      if datasets.test: 
        # This is only added so that the Test data can be visualized 
        summary_test, _ = session.run([merged, accuracy], feed_dict={ X_: datasets.test.data, y_: datasets.test.labels })
        if log_dir:
          test_writer.add_summary(summary_test, step)

  print("====\ndone learning\n====")
  if datasets.valid:
    summary_valid, acc_valid = session.run([merged, accuracy], feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels })
    print("Final validation accuracy: %.5f" % acc_valid)
    if log_dir:
      valid_writer.add_summary(summary_valid, num_steps)
 
  if datasets.test: 
    summary_test, loss_test, acc_test = session.run([merged, loss, accuracy], feed_dict={ X_: datasets.test.data, y_: datasets.test.labels })
    if log_dir:
      test_writer.add_summary(summary_test, num_steps)

    print("Test loss: %.5f" % loss_test)
    print("Test accuracy: %.5f" % acc_test)

  print("Num Epochs completed = %d " % datasets.train.epochs_completed)

  if save_model_dir:
    if not os.path.exists(save_model_dir):
      os.makedirs(save_model_dir)
    saver = tf.train.Saver()
    saver.save(session, save_model_dir)



if __name__ == "__main__":
  data_partition_fractions = [0.8, 0.1, 0.1]  # Train, Valid, Test

  # the input/output model to use
  #model = models.ModelWinningTeamOneHot(hot_streak=True, rival_streak=True)
  model = models.ModelSymmetrical(hot_streak=True, rival_streak=True)

  # the data model to use
  data_model = load_data.DATA_MODEL_COMPACT

  teamstats = team_stats.TeamStatsSeasonal()
  #teamstats = team_stats.TeamStatsPairMatchups()

  filenames = load_data.load_games(model, data_model, teamstats, num_historic_win_loss=10, normalize=False, save=True, num_splits=10)
  datasets = DataSetsFiles(filenames, data_partition_fractions, load_data.read_file, randomize=False)


  num_valid_examples = 0 if not datasets.valid else datasets.valid.num_examples
  num_test_examples = 0 if not datasets.test else datasets.test.num_examples
  print("example sizes: %d %d %d" % (datasets.train.num_examples, num_valid_examples, num_test_examples))

  save_model = './trained/lin-model/'
  train(datasets, model.log_dir, data_model.description, save_model_dir=save_model)



  # Below is testing if storing/restoring model works
  np.set_printoptions(threshold=np.nan)
  #print(np.cov(datasets.test.data.T))
  # Try reloading graph and recomputing test error
  saver = tf.train.Saver()
  with tf.Session() as session:
    meta_graph_location = save_model + ".meta"
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

    saver.restore(session, save_model)

    # Load required session vars
    X_ = tf.get_collection('X_')[0]
    y_ = tf.get_collection('y_')[0]
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
  
    # this data is not normalized....
    loss_, acc_ = session.run([loss, accuracy], feed_dict = { X_: datasets.test.data, y_: datasets.test.labels })


    print(loss_)
    print(acc_)
