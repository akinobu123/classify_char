import tensorflow as tf


class Model:

	def w_var (self, shape):
		initial = tf.truncated_normal (shape, stddev=0.1)
		return tf.Variable (initial)


	def b_var (self, shape):
		initial = tf.constant (0.1, shape=shape)
		return tf.Variable (initial)


	def conv (self, in_tensor, out_ch, kernel, strides, activation=tf.nn.relu):
		initializer = tf.random_normal_initializer(0, 0.02)
		out_tensor = tf.layers.conv2d(
			in_tensor, out_ch, kernel_size=[kernel,kernel], 
			strides=(strides,strides), padding="same", 
			kernel_initializer=initializer, 
			activation=activation, use_bias=True)
		return out_tensor


	def pool (self, in_tensor, pool_size):
		out_tensor = tf.layers.max_pooling2d (
			inputs=in_tensor, pool_size=[pool_size, pool_size], strides=pool_size)
		return out_tensor


	def fc (self, in_tensor, in_dim, out_ch):
		tensor = tf.matmul(in_tensor, self.w_var([in_dim, out_ch])) + self.b_var([out_ch])
		return tensor


	# for Mnist format
	def inference_28x28x1 (self, input_image, keep_prob, output_dim):
		# input_layer:   784 ==> 28 x 28 x 1
		input_layer = tf.reshape (input_image, [-1, 28, 28, 1])

		# block1: N x 28 x 28 x 1 ==> N x 14 x 14 x 64
		block1_layer = self.conv (input_layer,  64, 3, 1)	# in, out, kernel, stride
		block1_layer = self.conv (block1_layer, 64, 3, 1)	# in, out, kernel, stride
		block1_layer = self.pool (block1_layer, 2)			# in, poolsize

		# block2: N x 14 x 14 x 32 ==> N x 7 x 7 x 128
		block2_layer = self.conv (block1_layer, 128, 3, 1)	# in, out, kernel, stride
		block2_layer = self.conv (block2_layer, 128, 3, 1)	# in, out, kernel, stride
		block2_layer = self.pool (block2_layer, 2)			# in, poolsize

		# block3:  N x 7 x 7 x 128 ==> fc ==> 1024 ==> relu ==> dropout
		block3_tensor = tf.reshape (block2_layer, [-1, 7*7*128])
		block3_tensor = self.fc (block3_tensor, 7*7*128, 1024)
		block3_tensor = tf.nn.relu (block3_tensor)
		block3_tensor = tf.nn.dropout (block3_tensor, keep_prob)

		# output_layer: 1024 => fc => 10
		output_layer = self.fc (block3_tensor, 1024, output_dim)

		return output_layer


	# for 64 x 64 Grayscale format (N x 64 x 64 x 1)
	def inference_64x64x1 (self, input_image, keep_prob, output_dim):
		# input_layer:   N x 64 x 64 x 1
		input_layer = tf.reshape (input_image, [-1, 64, 64, 1])

		# block1: N x 64 x 64 x 1 ==> N x 32 x 32 x 64
		block1_layer = self.conv (input_layer,  128, 3, 1)	# in, out-ch, kernel, stride
		block1_layer = self.conv (block1_layer, 128, 3, 1)	# in, out-ch, kernel, stride
		block1_layer = self.pool (block1_layer, 2)			# in, poolsize

		# block2: N x 32 x 32 x 64 ==> N x 16 x 16 x 128
		block2_layer = self.conv (block1_layer, 256, 3, 1)	# in, out-ch, kernel, stride
		block2_layer = self.conv (block2_layer, 256, 3, 1)	# in, out-ch, kernel, stride
		block2_layer = self.pool (block2_layer, 2)			# in, poolsize

		# block3: N x 16 x 16 x 128 ==> N x 8 x 8 x 256
		block3_layer = self.conv (block2_layer, 512, 3, 1)	# in, out-ch, kernel, stride
		block3_layer = self.conv (block3_layer, 512, 3, 1)	# in, out-ch, kernel, stride
		block3_layer = self.conv (block3_layer, 512, 3, 1)	# in, out-ch, kernel, stride
		block3_layer = self.pool (block3_layer, 2)			# in, poolsize

		# block4:  N x 8 x 8 x 256 ==> fc ==> 4096 ==> relu ==> dropout
		block4_tensor = tf.reshape (block3_layer, [-1, 8*8*512])
		block4_tensor = self.fc (block4_tensor, 8*8*512, 4096)
		block4_tensor = tf.nn.relu (block4_tensor)
		block4_tensor = tf.nn.dropout (block4_tensor, keep_prob)

		# output_layer: 4096 => fc => output_dim
		output_layer = self.fc (block4_tensor, 4096, output_dim)

		return output_layer


	def train (self, output_label, correct_label):
		cross_entropy = tf.reduce_mean (
			tf.nn.softmax_cross_entropy_with_logits_v2 (
				labels=correct_label, logits=output_label))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		return train_step


	def accuracy (self, output_label, correct_label):
		correct_prediction = tf.equal(
			tf.argmax(output_label, 1), tf.argmax(correct_label, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def make_models (self, output_dim):
		# prepare place-holder
		images = tf.placeholder (tf.float32, shape=[None, 64, 64, 1])
		labels = tf.placeholder (tf.float32, shape=[None, output_dim])
#		images = tf.placeholder (tf.float32, shape=[None, 784]) # 28 x 28
#		labels = tf.placeholder (tf.float32, shape=[None, output_dim])
		keep   = tf.placeholder (tf.float32)
		# make NN models
		infer = self.inference_64x64x1 (images, keep, output_dim)
#		infer = self.inference_28x28x1 (images, keep, output_dim)
		train = self.train (infer, labels)
		accuracy = self.accuracy (infer, labels)
		return infer, train, accuracy, images, labels, keep

