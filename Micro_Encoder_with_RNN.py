from __future__ import division
from __future__ import print_function

#from microtext import microtext
import numpy as np
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import os.path
import os  
import random
from tensorflow.python.framework import dtypes
# abbre_len = 25
# words_len = 15
# words_len2 = 10
# global abbre_len
# global words_len2 
batch_size = 128
test_batch_num = 1
####
def load_train_data():
	char_set = []
	words_set = []
	train_x = []
	train_y = []
	train_x_len = []
	train_y_word_len = []
	train_char_y = []
	for key, value in microtext.items():
		### consider there is only one word in the abbre
 
		if len(value[0].strip().split('_'))>=0: 
			char_set.extend([char for char in key if char not in char_set])
			char_set.extend([char for char in value[0] if char not in char_set])
			words_set.extend([word for word in value[0].strip().split('_')  if word not in words_set])
	 		x_Data = [char_set.index(char)+1 for char in key]
	 		y_Data = [words_set.index(word)+1 for word in  value[0].strip().split('_')]
	 		x_Data_len = len(value[0])
	 		y_Data_len = len(value[0].strip().split('_'))
	 		y_char_Data = [char_set.index(char) for char in value[0]]
 			train_y.append(y_Data)
 			train_x.append(x_Data)
 			train_x_len.append(x_Data_len)
 			train_y_word_len.append(y_Data_len)
 			train_char_y.append(y_char_Data)
 	abbre_len = max(train_x_len)
 	words_len2 = max(train_y_word_len)
 
 	train_x = sequence.pad_sequences(train_x, maxlen = abbre_len, padding='pre')
 	train_y = sequence.pad_sequences(train_y, maxlen = words_len2, padding='pre')
 	# train_char_y = sequence.pad_sequences(train_char_y, maxlen = words_len, padding='pre')	
 	print(np.array(train_x).shape, np.array(train_y).shape)
 
	print ("Please set word_set_len to %s and char_set_len to %s abbre_len to %s and words_len2 to %s"%(len(words_set),len(char_set), abbre_len, words_len2))

 	return train_x, train_y, train_x_len, train_y_word_len
def load_train_data1():
	filname = "Microtext Latest database (without polarity and category).csv"
	lines = open(filname,'r').readlines()
	#lines = open('/home/ranjan/Documents/cicling/microtext-code/Dataset_Normalization/Text_Norm_Data_Release_Fei_Liu.txt','r').readlines()
	char_set = []
	words_set = []
	train_x = []
	train_y = []
	train_x_len = []
	train_y_word_len = []
	train_char_y = []
	for line in lines:
		### consider there is only one word in the abbre
		if filname=="Text_Norm_Data_Release_Fei_Liu.txt":
			line = line.strip().split('\t')[1]
	 		key = line.strip().split(' | ')[0]
	 		value = line.strip().split(' | ')[1] 			
	 	elif filname=="microtext_english.txt" or filname == "Microtext Latest database (without polarity and category).csv": 
 	 		key = line.strip().split('\t')[0]
	 		value = line.strip().split('\t')[1] 

		char_set.extend([char for char in key if char not in char_set])
		char_set.extend([char for char in value if char not in char_set])
		words_set.extend([word for word in value.strip().split(' ')  if word not in words_set])
	 	x_Data = [char_set.index(char)+1 for char in key]
	 	y_Data = [words_set.index(word)+1 for word in  value.strip().split(' ')]
	 	x_Data_len = len(key)
	 	y_Data_len = len(value.strip().split(' '))
	 	y_char_Data = [char_set.index(char) for char in value]
 		train_y.append(y_Data)
 		train_x.append(x_Data)
 		train_x_len.append(x_Data_len)
 		train_y_word_len.append(y_Data_len)
 		train_char_y.append(y_char_Data)
 	abbre_len = max(train_x_len)
 	words_len2 = max(train_y_word_len)
 
 	train_x = sequence.pad_sequences(train_x, maxlen = abbre_len, padding='pre')
 	train_y = sequence.pad_sequences(train_y, maxlen = words_len2, padding='pre')
 	# train_char_y = sequence.pad_sequences(train_char_y, maxlen = words_len, padding='pre')	
 	print("#####################################\n#####################################")
 	print (" Please set word_set_len to %s and \n char_set_len to %s abbre_len to %s \n and words_len2 to %s"%(len(words_set),len(char_set), abbre_len, words_len2))
	print("#####################################\n#####################################")

 	return train_x, train_y, train_x_len, train_y_word_len, char_set, words_set
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)
class Abbre_Repair(object):
	"""docstring for VAE"""
	def __init__(self):	
		self.params = []
		self.latent_dim = 256
		self.word_set_len = 3380+1#1564 +1#
		self.char_set_len =  82+1 #97 +1
		
		self.abbre_len = 31
		self.words_len2 = 15

		self.x = tf.placeholder(tf.int32, [None, self.abbre_len], name="input_x")
		 
		self.y = tf.placeholder(tf.float32, [None, self.words_len2], name="input_y")
		self.embedding_char = tf.Variable(tf.random_uniform([self.abbre_len, 50], -1.0, 1.0), name='embedding_ichar', trainable=True)
		# self.char_y = tf.placeholder(tf.float32, [None, words_len], name = "input_char_y")
		self.x_len  = tf.placeholder(tf.int32, [None], name = "train_x_len")
		self.y_len = tf.placeholder(tf.int32, [None], name = "train_y_len")
 
		self.l2_loss = tf.constant(0.0)
		x = tf.reshape(tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.char_set_len, 1.0, 0.0), [-1, self.char_set_len, self.abbre_len])
		# y_char = tf.reshape(tf.one_hot(tf.to_int32(tf.reshape(self.char_y, [-1])), self.char_set_len, 1.0, 0.0), [-1, self.words_len*self.char_set_len])
 
		encoder = self.Encoder(x) 
		decoder = self.Decoder(encoder) 
		# for k in range(words_len2):
		yy = tf.reshape(tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.word_set_len, 1.0, 0.0), [-1, self.words_len2, self.word_set_len]) 
		yy_hat = tf.cast(tf.reshape(decoder, [-1, self.words_len2, self.word_set_len]), tf.float32)
		print ("yy_hat", yy_hat)
		self.out = tf.argmax(yy_hat, 2)
		mask = tf.cast(tf.sign(self.y), tf.float32)
		self.mask = mask
		self.test = (tf.argmax(yy, 2))
		self.test1 = tf.argmax(yy_hat, 2)

 		correct_pred = tf.equal(tf.argmax(yy, 2), tf.argmax(yy_hat, 2)) 
 		# correct_pred = tf.transpose(correct_pred, [1,0])
		self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
		# self.acc = tf.metrics.accuracy(tf.argmax(yy, 2), tf.argmax(yy_hat, 2), tf.transpose(mask,[1,0]))
		# yy_logits = tf.reshape(yy_hat, [-1, words_len2, self.word_set_len])
		# self.loss = tf.contrib.seq2seq.sequence_loss(yy_logits, self.y, tf.ones([batch_size,words_len2]),average_across_timesteps=True, average_across_batch=True)
		# self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = yy, logits = yy_hat))# + 0.001*self.l2_loss
		# self.decoder = decoder
		cross_entropy = yy * tf.log(yy_hat)
		cross_entropy = -tf.reduce_sum(cross_entropy, 2)
		# cross_entropy =tf.transpose(cross_entropy, [1, 0])

		cross_entropy *= mask
		cross_entropy = tf.reduce_mean(cross_entropy) 
		self.loss = cross_entropy #+ 0.5*self.l2_loss
 

		# self.acc = accs/words_len2
		# self.loss = self.loss/words_len2
	def Encoder(self, x):
		with tf.name_scope("encoder"): 
			char_embedding = x#tf.nn.embedding_lookup(self.embedding_char, x)
			num_hidden = self.latent_dim
			#cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
			cell = tf.contrib.rnn.GRUCell(num_hidden)
			val, state = tf.nn.dynamic_rnn(cell, char_embedding, dtype=tf.float32, sequence_length=self.x_len)
			# # val, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = x, dtype=tf.float32, sequence_length=self.x_len)
			# val = tf.transpose(val, [1, 0, 2])  
			# # val_bw = tf.transpose(val[1], [1, 0, 2])
			# val = tf.gather(val, int(val.get_shape()[0]) - 1)
			# # val = tf.concat([val[0], val[1]], 2)
			# W_encoder = weight_variable([num_hidden, self.latent_dim])
			# b_encoder = bias_variable([self.latent_dim])
			# self.params.append(W_encoder)
			# self.params.append(b_encoder)

			# self.l2_loss += tf.nn.l2_loss(W_encoder)
 
			# encoder = tf.tanh(tf.nn.xw_plus_b(val, W_encoder, b_encoder))
  
		return val
	def Decoder(self, encoder):
		with tf.name_scope("encoder"): 
			out = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.words_len2,
                                             dynamic_size=False, infer_shape=True)
			# encoder = tf.reshape(encoder, [-1, self.latent_dim])
			cell_decode = tf.contrib.rnn.LSTMCell(self.latent_dim, state_is_tuple = True)
			cell_decode = tf.contrib.rnn.GRUCell(self.latent_dim)
			val, state = tf.nn.dynamic_rnn(cell_decode, encoder, dtype=tf.float32, sequence_length = self.y_len, scope="decoder")
			print (encoder,"en")
			val = tf.transpose(encoder, [1, 0, 2]) 
 
			# encoder = tf.transpose(encoder,[1,0,2])
			# print ("encoder",encoder)
			# helper = tf.contrib.seq2seq.TrainingHelper(encoder, self.y_len, time_major= True) 
			# my_decoder = tf.contrib.seq2seq.BasicDecoder(cell_decode, helper, initial_state=cell_decode.zero_state(dtype=dtypes.float32, batch_size=batch_size)) 
			# outputs, final_context_state = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=True,swap_memory=True, scope="decoder")

			# out = outputs.rnn_output
			# print("out", out)
			for k in range(self.words_len2):
				 
				val_f = tf.gather(val, int(val.get_shape()[0]) + k - self.words_len2)
				W_encoder1 = weight_variable([self.latent_dim, 1024])

				b_encoder1 = bias_variable([1024])
				self.params.append(W_encoder1)
				self.params.append(b_encoder1)

				self.l2_loss += tf.nn.l2_loss(W_encoder1)
				decoder1 = tf.nn.relu(tf.nn.xw_plus_b(val_f, W_encoder1, b_encoder1))		 


				W_encoder = weight_variable([1024, self.word_set_len])

				b_encoder = bias_variable([self.word_set_len])
				self.params.append(W_encoder)
				self.params.append(b_encoder)

				self.l2_loss += tf.nn.l2_loss(W_encoder)
				decoder = tf.nn.relu(tf.nn.xw_plus_b(decoder1, W_encoder, b_encoder))
				decoder = tf.nn.softmax(decoder)
				out = out.write(k, decoder)

 
        	out = tf.transpose(out.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
  
		return decoder 
 
  
def main(): 
	model = Abbre_Repair()
	params = model.params  
	train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss) 
	train_x, train_word_y, train_x_len, train_y_word_len, char_set, words_set = load_train_data1()
	abbre_len = max(train_x_len)
	words_len2 = max(train_y_word_len)
 
	saver = tf.train.Saver()
 
	config_gpu = tf.ConfigProto() 
	# config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.4 
	with tf.Session(config=config_gpu) as sess:
	 
		sess.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state('./save')
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successful restore from previous models")
		else:
			print ("There is no pretrained model")
		summary_writer_train = tf.summary.FileWriter('./save/train',graph=sess.graph)
		summary_writer_test = tf.summary.FileWriter('./save/test') 
		epoches = 2000 
		
		batches_num = int(len(train_x)/batch_size)
		# test_batches = range(batches_num)[2:10]#
		test_batches = random.sample(range(batches_num),test_batch_num)
		train_batches = [num for num in range(batches_num) if num not in test_batches]

 
		for epo in range(epoches):
			train_Loss = []
			train_acc = [] 
			
			for step in (train_batches):
				batch_x = train_x[step*(batch_size):(step+1)*batch_size]
				batch_y = train_word_y[step*batch_size : (step+1)*batch_size]
				batch_x_len = train_x_len[step*batch_size : (step+1)*batch_size]
				batch_y_word_len = train_y_word_len[step*batch_size : (step+1)*batch_size]

				feed_dict = {model.x: batch_x,  model.y: batch_y, model.x_len: batch_x_len, model.y_len: batch_y_word_len}
				# print (batch_x)
				_, loss, acc, mask, mask1, mak = sess.run([train_step, model.loss, model.acc, model.test, model.test1, model.mask], feed_dict=feed_dict)
				train_Loss.append(loss)
				train_acc.append(acc)					
  				# print (mask[-1],mask1[-1])
  				# print (mak[-1])
			# print("Train epo {0}| CONS: {1} | ACC: {2}".format(epo , np.mean(train_Loss), np.mean(train_acc)))

			# save_path = saver.save(sess, "./save/model.ckpt")
			test_Loss= []
			test_acc = []
			for step in test_batches: 
				batch_x = train_x[step*(batch_size):(step+1)*batch_size]
				batch_y = train_word_y[step*batch_size : (step+1)*batch_size]
				batch_x_len = train_x_len[step*batch_size : (step+1)*batch_size]
				batch_y_word_len = train_y_word_len[step*batch_size : (step+1)*batch_size]

				feed_dict = {model.x: batch_x,  model.y: batch_y, model.x_len: batch_x_len, model.y_len: batch_y_word_len}
				loss, acc, samples = sess.run([model.loss, model.acc, model.out], feed_dict=feed_dict)
				test_Loss.append(loss)
				test_acc.append(acc)
				sample = random.randint(0, len(samples)-1)	
				input_index = batch_x[sample]
				output_index = samples[sample]
				# print(input_index, output_index)
				 
				input_txt = [char_set[int(k)-1] for k in input_index if k!=0 ]
				input_char = ''
				for char in input_txt:
					input_char+=char

				output_txt = [words_set[int(k)-1] for k in output_index][0]
				sample = str(input_char)+ "-->"+ str(output_txt) 
			print("Epo %s Train: Loss %.4f ACC %.4f | Test: Loss %.4f ACC %.4f, sample: %s"%(epo , np.mean(train_Loss), np.mean(train_acc), np.mean(test_Loss), np.mean(test_acc),sample))


main()