from __future__ import division

import tensorflow as tf
import sys

def variable_summaries(var,var_name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var_name+'_summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class Model(object):

	def __init__(self,features,labels,seq_length,config,is_training):

		batch_size=1


		# lstm cells definition
		with tf.variable_scope('forward'):

			forward_cells = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_forward = tf.contrib.rnn.LSTMCell(config.n_hidden,use_peepholes=False,
                                            forget_bias=1.0,activation=tf.tanh,
                                            initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
					forward_cells.append(lstm_cell_forward)

			cells = tf.nn.rnn_cell.MultiRNNCell(forward_cells)
			init_states = cells.zero_state(batch_size,dtype=tf.float32)

		with tf.variable_scope('RNN'):
			rnn_outputs, output_state_fw = tf.nn.dynamic_rnn(
											cell=cells,
											inputs=features,
											initial_state=init_states,
											dtype=tf.float32,
											sequence_length=seq_length)

			self._out_before_slice=rnn_outputs
			rnn_output = tf.slice(rnn_outputs,[0,tf.shape(rnn_outputs)[1]-1,0],[-1,-1,-1])
			self._out_after_slice=rnn_output

		with tf.variable_scope('output'):
			output_fw_weights = tf.get_variable('forward_weights',[config.n_hidden,config.audio_labels_dim],dtype=tf.float32,
			                        initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases = tf.get_variable('biases',shape=[config.audio_labels_dim],dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			rnn_output = tf.reshape(rnn_output,[-1,config.n_hidden])	
			
			output = tf.matmul(rnn_output,output_fw_weights) + output_biases
		
			logits = tf.reshape(output,[batch_size,-1,config.audio_labels_dim])

		posteriors=tf.nn.softmax(logits,name='SMO')
		prediction=tf.argmax(logits, axis=2)
		correct = tf.equal(prediction,tf.to_int64(labels))
		accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

		self._posteriors=posteriors
		self._accuracy=accuracy
		self._labels = labels
		self._prediction = prediction

	@property
	def out_before_slice(self):
		return self._out_before_slice


	@property
	def out_after_slice(self):
		return self._out_after_slice
		
	@property
	def cost(self):
		return self._cost

	@property
	def optimize(self):
		return self._optimize

	@property
	def posteriors(self):
		return self._posteriors

	@property
	def correct(self):
		return self._correct

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def labels(self):
		return self._labels

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def global_step(self):
		return self._global_step


	@property
	def prediction(self):
		return self._prediction