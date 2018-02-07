def bidirectional_rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  cell_fw = tf.nn.rnn_cell.GRUCell(num_units=EMBEDDING_SIZE)
  cell_bw = tf.nn.rnn_cell.GRUCell(num_units=EMBEDDING_SIZE)

  outputs, encoding = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        dtype=tf.float32,
        sequence_length=features[FEATURE_LEN],
        inputs=word_vectors)

  cell_fw_state, cell_bw_state = encoding
  encoding = tf.concat(axis=1, values=[cell_fw_state, cell_bw_state])
  dense0 = tf.layers.dense(encoding, units=512, activation=tf.nn.relu)   
  logits = tf.layers.dense(dense0, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)

def multi_layer_rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  dropout = 0.2
  num_layers = 3
  cells = []
  for _ in range(num_layers):
      cell = tf.contrib.rnn.GRUCell(num_units=EMBEDDING_SIZE)  
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
      cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=False)   
  outputs, encoding = tf.nn.dynamic_rnn(
        cell,
        dtype=tf.float32,
        sequence_length=features[FEATURE_LEN],
        inputs=word_vectors)
    
  encoding = array_ops.reshape(encoding, [-1, num_layers, EMBEDDING_SIZE])
  encoding = encoding[:, -1, :]
      
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)
