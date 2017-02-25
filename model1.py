import math
import numpy as np
import tensorflow as tf

# User defined modules
import load_data
from data_sets import DataSetsFiles
from flag_values import _FlagValues as FlagValues

hyperparameters = {
  'num_historic_win_loss': 10,
  'learn_rate': 0.005,
  'hidden_layer1_size': 1024
}

FLAGS = FlagValues()
FLAGS.learning_rate = 1e-4
FLAGS.summaries_dir = './logs'



#dataset, labels = load_data.load_games(detailed=True, include_tourney = True, num_historic_win_loss = 10, aggregate_all_data = True)

#print("datasets memory = %d" % dataset.nbytes)
data_partition_fractions = [0.8, 0.1, 0.1]  # Train, Valid, Test

filenames= load_data.load_games(detailed=True, include_tourney=True, num_historic_win_loss=10, aggregate_all_data=True, normalize=True, save=True, num_splits=10)
datasets = DataSetsFiles(filenames, data_partition_fractions, load_data.read_file)


#del dataset

if __name__ == "__main__":
  print("1")
  # Hyper Parameters
  hidden_layer1_size = 1024


  num_input_features = datasets.num_features
  num_classification_labels = datasets.num_classification_labels # number of unique teams


  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

  def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

  def bias_variable(shape, const=0.1):
    initial = tf.constant(const, shape=shape)
    return tf.Variable(initial)

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
    'out': hidden_layer1_size
  }
    
  layer2 = {
    'in' : layer1['out'],
    'out': num_classification_labels
  }

  X_ = tf.placeholder(tf.float32, shape=(None, layer1['in']))
  y_ = tf.placeholder(tf.float32, shape=(None, num_classification_labels))

  keep_prob = tf.placeholder(tf.float32)  # dropout for hidden layer
  

  # Model
  # Variables.
  weights_layer1 = weight_variable([layer1['in'], layer1['out']])
  biases_layer1 = bias_variable([layer1['out']])

  weights_layer2 = weight_variable([layer2['in'], layer2['out']], stddev=math.sqrt(2.0/hidden_layer1_size))
  biases_layer2 = bias_variable([layer2['out']])



  # Layers
  hidden1 = tf.nn.dropout(nn_layer(X_, weights_layer1, biases_layer1, 'hidden1'), keep_prob)

  # Do not apply softmax yet
  output_activation = nn_layer(hidden1, weights_layer2, biases_layer2, 'output_activation')

  # Training computation.
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_activation, labels=y_))

  # TensorBoard
  tf.summary.scalar('cross_entropy', loss)

  # Optimizer.
  #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

  with tf.name_scope('accuracy'): 
    output_layer = tf.nn.softmax(output_activation) 
    # Predictions for the training, validation, and test data.
    predictions = tf.argmax(output_layer, 1)
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(predictions, tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #accuracy = tf.scalar_mul(100, tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
  tf.summary.scalar('accuracy', accuracy)

  # Merge summaries
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', session.graph)
  valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/valid')
  test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
                                     
  #####
  batch_size = 64
  num_steps = math.ceil(datasets.train.num_examples / batch_size * 20)
  print("Num steps = %d" % num_steps)
    
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(1, num_steps):
    # Generate minibatch.
    batch_data, batch_labels = datasets.train.next_batch(batch_size)

    feed_dict = {X_ : batch_data, y_ : batch_labels, keep_prob: 0.5 }
    summary, loss_, _, acc = session.run([merged, loss, optimizer, accuracy], feed_dict=feed_dict)

    train_writer.add_summary(summary, step)

    if (step % 1000 == 0):
      print("Minibatch loss at step %d: %f" % (step, loss_))
      print("Minibatch accuracy: %.3f" % acc) #accuracy.eval(feed_dict={X_: batch_data, y_: batch_labels, keep_prob: 1.0 }))
      
      summary_valid, acc_valid = session.run([merged, accuracy], feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels, keep_prob: 1.0 })
      valid_writer.add_summary(summary_valid, step)
      print("Validation accuracy: %.3f" % acc_valid)#accuracy.eval(feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels, keep_prob: 1.0 }))

  summary_valid, acc_valid = session.run([merged, accuracy], feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels, keep_prob: 1.0 })
  valid_writer.add_summary(summary_valid, num_steps)
  print("Final validation accuracy: %.3f" % acc_valid)#accuracy.eval(feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels, keep_prob: 1.0 }))
  
  summary_test, acc_test = session.run([merged, accuracy], feed_dict={ X_: datasets.valid.data, y_: datasets.valid.labels, keep_prob: 1.0 })
  test_writer.add_summary(summary_test, num_steps)
  print("Test accuracy: %.3f" % acc_test)#accuracy.eval(feed_dict={ X_: datasets.test.data, y_: datasets.test.labels, keep_prob: 1.0 }))

  print("Num Epochs completed = %d " % datasets.train.epochs_completed)
