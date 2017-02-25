import numpy as np

def randomize_data(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation, :]
  return shuffled_dataset, shuffled_labels


class DataSetsFiles:
  def __init__(self, filenames, data_partition_fractions, read_file_fn, randomize=True):
    num_files = len(filenames)
    train_end_index = int(data_partition_fractions[0] * num_files)
    valid_end_index = int((data_partition_fractions[0] + data_partition_fractions[1]) * num_files)
    test_end_index = int(sum(data_partition_fractions) * num_files)

    if randomize:
      np.random.shuffle(filenames)
      
    # Create datasets, Train, Valid, Test
    self._train = DataSetFiles(filenames[:train_end_index], read_file_fn)
    self._valid = DataSetFiles(filenames[train_end_index:valid_end_index], read_file_fn)
    self._test = DataSetFiles(filenames[valid_end_index:test_end_index], read_file_fn)

  @property
  def num_features(self):
    return self._valid.num_features

  @property
  def num_classification_labels(self):
    return self._valid.num_classification_labels

  @property
  def train(self):
    return self._train

  @property
  def valid(self):
    return self._valid

  @property
  def test(self):
    return self._test

class DataSetFiles:
  def __init__(self, filenames, read_file_fn):
    self._filenames = filenames
    np.random.shuffle(self._filenames)

    print("num files = %d" % len(self._filenames))
    self._num_examples = 0
    for f in filenames:
      _, labels = read_file_fn(f)
      self._num_examples += labels.shape[0]

    print("num examples = %d" % self._num_examples)
    
    self._read_file = read_file_fn

    self._current_file_index = -1
    self.next_file()

    self._file_epochs_completed = 0
    self._mini_epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def num_features(self):
    return self._data.shape[1]

  @property
  def num_classification_labels(self):
    return self._labels.shape[1]

  @property
  def epochs_completed(self):
    print("%d %d" % (self._mini_epochs_completed, self._file_epochs_completed))
    if self._file_epochs_completed > 0:
      return self._mini_epochs_completed / self._file_epochs_completed
    else:
      return self._mini_epochs_completed


  def read_file(self, filename):
    return self._read_file(filename)

  def next_file(self):
    # Go to next file
    self._current_file_index += 1
    if self._current_file_index >= len(self._filenames):
      # Shuffle files and restart
      np.random.shuffle(self._filenames)
      self._current_file_index = 0
      
      # Finished epoch
      self._file_epochs_completed += 1
    
    self._data, self._labels = self.read_file(self._filenames[self._current_file_index])
    self._current_num_examples = self._data.shape[0]
    
    
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._current_num_examples:
      #Finished mini file epoch
      self._mini_epochs_completed += 1

      # Update file
      self.next_file()
      
      # Shuffle the data
      perm = np.arange(self._current_num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._labels = self._labels[perm]

      # Start next epoch
      self._index_in_epoch = 0
      return self._data[start:], self._labels[start:]
    else:
      end = self._index_in_epoch
      return self._data[start:end], self._labels[start:end]

class DataSets:
  def __init__(self, data, labels, data_partition_fractions, normalize=True, randomize=True):

    assert(len(data.shape) == 2)  # Num Examples x Num Features

    train_end_index = int(data_partition_fractions[0]*data.shape[0])
    valid_end_index = int((data_partition_fractions[0] + data_partition_fractions[1])*data.shape[0])
    test_end_index = int(sum(data_partition_fractions) * data.shape[0])

    if normalize:
      # Normalize all data together
      data = data.astype(np.float32)
      mu = np.tile(np.mean(data, axis=0), (data.shape[0], 1))
      data = (data - mu) / (np.max(data, axis=0) - np.min(data, axis=0))
      data = np.nan_to_num(data)  # Convert NaNs to 0


    if randomize:
      data, labels = randomize_data(data, labels)
      
    # Create datasets, Train, Valid, Test
    self._train = DataSet(data[:train_end_index, :], labels[:train_end_index], normalize=False)
    self._valid = DataSet(data[train_end_index:valid_end_index], labels[train_end_index:valid_end_index], normalize=False)
    self._test = DataSet(data[valid_end_index:test_end_index], labels[valid_end_index:test_end_index], normalize=False)

  @property
  def num_features(self):
    return self._valid.num_features

  @property
  def num_classification_labels(self):
    return self._valid.num_classification_labels

  @property
  def train(self):
    return self._train

  @property
  def valid(self):
    return self._valid

  @property
  def test(self):
    return self._test

class DataSet(object):
  def __init__(self, data, labels, normalize=True):
    assert data.shape[0] == labels.shape[0], (
        "data.shape: %s labels.shape: %s" % (data.shape,
                                               labels.shape))
    self._num_examples = data.shape[0]
    assert(len(data.shape) == 2) # Assert shape is (num_examples, num_features)
    
    if normalize:
      # Convert features (in columns) to have mean=0, stddev=1
      print("TODO: Not Implemented")

    self._data = np.array(data)
    self._labels = np.array(labels)
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def num_features(self):
    return self._data.shape[1]

  @property
  def num_classification_labels(self):
    return self._labels.shape[1]

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert(batch_size <= self._num_examples)
    end = self._index_in_epoch
    to_return = self._data[start:end], self._labels[start:end]
    return to_return
