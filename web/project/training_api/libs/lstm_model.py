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

	def __init__(self, nclasses, features, labels, seq_length, model_data, is_training):

		if is_training:
			batch_size = model_data['batch_size']
		else:
			batch_size = 1

		global_step = tf.Variable(0, trainable=False)
		self._global_step = global_step

		# lstm cells definition
		with tf.variable_scope('forward'):

			forward_cells = []
			for i in range(model_data['num_layers']):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_forward = tf.contrib.rnn.LSTMCell(model_data['n_hidden'],use_peepholes=False,
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
			output_fw_weights 	= tf.get_variable('forward_weights',[model_data['n_hidden'], nclasses],dtype=tf.float32,
			                        			initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases 		= tf.get_variable('biases',shape=[nclasses],dtype=tf.float32,
												initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			rnn_output 	= tf.reshape(rnn_output,[-1,model_data['n_hidden']])	
			output 		= tf.matmul(rnn_output,output_fw_weights) + output_biases
			logits 		= tf.reshape(output,[batch_size,-1, nclasses])

		if is_training:
		# evaluate cost and optimize
			with tf.name_scope('cost'):
				self._cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
				tf.summary.scalar('cost',self._cost)

			with tf.name_scope('optimizer'):
				#learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
			    #                    config.updating_step, config.learning_decay, staircase=True)

				learning_rate = model_data['learning_rate']
				self._learning_rate= learning_rate

				if "momentum" in model_data['optimizer']:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
				elif "adam" in model_data['optimizer']:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
				else:
					print("Optimizer must be either momentum or adam. Closing.")
					raise ValueError("Optimizer must be either momentum or adam. Closing.")

				# gradient clipping
				gradients , variables = zip(*self._optimizer.compute_gradients(self._cost))
				clip_grad  = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients] 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)


		posteriors 	= tf.nn.softmax(logits,name='SMO')
		prediction 	= tf.argmax(logits, axis=2)
		correct 	= tf.equal(prediction,tf.to_int64(labels))
		accuracy 	= tf.reduce_mean(tf.cast(correct,tf.float32))

		self._posteriors	= posteriors
		self._accuracy		= accuracy
		self._labels 		= labels
		self._prediction 	= prediction

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