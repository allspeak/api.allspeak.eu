from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from . import utilities
from . import lstm_model

# Avoid printing tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# ============================================================================
# Training routine
# ============================================================================
def trainPureUserLSTM(training_data_len, model_data, output_net_name, session_path, clean_folder):

	# define data paths
	data_path 		= os.path.join(session_path, 'data')

    # define temp folder to create NET files
    train_output_net_path = os.path.join(session_path, 'data', 'net')  # instance/users_data/APIKEY/train_data/training_sessionid/data/net  

	# create them if !exist
	if not os.path.exists(train_output_net_path):
		os.makedirs(train_output_net_path)

	trainingLogFile = open(os.path.join(session_path, 'train_log.txt'), 'w')

	# list of input filenames + check existence
	filename_train = [(data_path + '/sequence_full_{:04d}.tfrecords'.format(i)) for i in range(training_data_len)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# training graph		
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			with tf.name_scope('train_batch'):
				audio_features, audio_labels, seq_length = utilities.input_pipeline(filename_train, model_data)
		
		# audio features reconstruction
		with tf.device('/cpu:0'):
			with tf.variable_scope('model',reuse=None):
				print('Building training model:')
				train_model = lstm_model.Model(audio_features, audio_labels, seq_length, model_data, is_training=True)
				print('done.\n')
		
		# variables initializer
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=10)

		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:
			# run initializer
			sess.run(init_op)

			# start queue coordinator and runners
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			print('## optimizer : ', model_data.optimizer)
			trainingLogFile.write('## optimizer : {:s} \n'.format(model_data.optimizer"))
			
			print('## number of hidden layers : ',model_data.num_layers)
			trainingLogFile.write('## number of hidden layers : {:d} \n'.format(model_data.num_layers))
			
			print('## number of hidden units : ',model_data.n_hidden)
			trainingLogFile.write('## number of hidden units : {:d} \n'.format(model_data.n_hidden))
			
			print('## learning rate : ',model_data.learning_rate)
			trainingLogFile.write('## learning rate : {:.6f} \n'.format(model_data.learning_rate))
			
			print('## batch size : ',model_data.batch_size)
			trainingLogFile.write('## batch size : {:d} \n'.format(model_data.batch_size))
			
			print('## number of steps: ',training_data_len*model_data.num_epochs/model_data.batch_size)
			trainingLogFile.write('## approx number of steps: {:d} \n'.format(int(training_data_len*model_data.num_epochs/model_data.batch_size)))
			
			print('## number of steps per epoch: ',training_data_len/model_data.batch_size)
			trainingLogFile.write('## approx number of steps per epoch: {:d} \n'.format(int(training_data_len/model_data.batch_size)))
			print('')

			try:
				epoch_counter=1
				epoch_cost=0.0
				EpochStartTime=time.time()

				step=1
				while not coord.should_stop():
					_ , C  , train_pred , train_label ,\
					out_before_slice, out_after_slice  = sess.run([ train_model.optimize,
																	train_model.cost,
																	train_model.prediction,
																	train_model.labels,
																	train_model.out_before_slice,
																	train_model.out_after_slice])
					epoch_cost += C

					if (step % 50 == 0 or step==1):
						print("step[{:7d}] cost[{:2.5f}] ".format(step,C))

					if ((step % int(num_examples / .batch_size) == 0) and (step is not 0)):

						# at each step we get the average cost over a batch, so divide by the 
						# number of batches in one epoch
						epoch_cost /=  (num_examples/model_data.batch_size)

						print('Completed epoch {:d} at step {:d} --> cost[{:.6f}]'.format(epoch_counter, step, epoch_cost))
						print('Epoch training time (seconds) = ',time.time() - EpochStartTime)
						
						out_every_epoch=1
						if((epoch_counter%out_every_epoch)==0):
							save_path = saver.save(sess, os.path.join(train_output_net_path, 'model_epoch' + str(epoch_counter) + '.ckpt'))

						epoch_counter	+=1
						epoch_cost		=0.0
						EpochStartTime	=time.time()

					step += 1

			except tf.errors.OutOfRangeError:
				print('---- Done Training: epoch limit reached ----')
			finally:
				coord.request_stop()

			coord.join(threads)

			save_path = saver.save(sess, os.path.join(train_output_net_path, 'model_end.ckpt'))
			print("model saved in file: %s" % save_path)

            result = freeze.freeze(output_model_name, train_output_net_path, sOutputNodeName)

	trainingLogFile.close()
    # move optimized net to session path
    os.rename(result['grp_opt_name'], os.path.join(session_path, 'opt' + output_model_name + '.pb'))
    # delete other files
    if clean_folder is True and os.path.isdir(os.path.join(session_path, 'data')) is True:
        shutil.rmtree(os.path.join(session_path, 'data'))



