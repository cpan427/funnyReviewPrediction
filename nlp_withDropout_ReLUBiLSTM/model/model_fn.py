"""Define the model."""

import tensorflow as tf
import numpy as np


def build_model(mode, inputs, params, is_training):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    sentence = inputs['sentence']

    if params.model_version == 'lstm':
        # Get word embeddings for each token in the sentence
        
        # Get word embeddings for each token in the sentence
        embed = np.genfromtxt("/afs/.ir.stanford.edu/users/c/a/capan/CS224N/finalProject/funnyReviewPrediction/nlp_GG2/data/small/embeddings.csv", delimiter=',')
        embeddings = tf.constant(embed, name="embeddings", dtype = tf.float32)
        sentence = tf.nn.embedding_lookup(embeddings, sentence)
        
        sentence = tf.layers.dense(sentence, 25, activation = tf.nn.relu)
        
        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        outputs, last_output_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell, lstm_cell, sentence, dtype=tf.float32, sequence_length=inputs['sentence_lengths'])
        
        
        
        output = tf.concat(outputs, 2)
        meanOutput = tf.reduce_mean(output, axis = 1)
        
        # Compute logits from the output of the LSTM
        layer1_output = tf.layers.dense(meanOutput, 25, activation = tf.nn.relu)
        if is_training:
            layer1_output = tf.nn.dropout(layer1_output, 1 - params.dropout_rate)
        
        layer2_output = tf.layers.dense(layer1_output, 25, activation = tf.nn.relu)
        if is_training:
            layer2_output = tf.nn.dropout(layer2_output, 1 - params.dropout_rate)
        
        layer3_output = tf.layers.dense(layer2_output, 25, activation = tf.nn.relu)
        if is_training:
            layer3_output = tf.nn.dropout(layer3_output, 1 - params.dropout_rate)
        
        layer4_output = tf.layers.dense(layer3_output, 25, activation = tf.nn.relu)
        if is_training:
            layer4_output = tf.nn.dropout(layer4_output, 1 - params.dropout_rate)
        
        layer5_output = tf.layers.dense(layer4_output, 25, activation = tf.nn.relu)
        if is_training:
            layer5_output = tf.nn.dropout(layer5_output, 1 - params.dropout_rate)
        
        logits = tf.layers.dense(layer5_output, 2)

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    sentence_lengths = inputs['sentence_lengths']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params, is_training)
        predictions = tf.cast(tf.argmax(logits, -1), tf.int32)
        #labels = tf.reshape(labels, [-1,1])
        #predictions = [1 for i in logits > 0 else 0]

    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(losses)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
