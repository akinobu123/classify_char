import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Supply:
	mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

	def get_output_dim(self):
		return 10

	def next_train_batch (self, batch_size):
		batch =  self.mnist.train.next_batch (batch_size)
		return batch[0], batch[1]

	def get_test_data (self):
		test_datas = self.mnist.test.images
		test_labels = self.mnist.test.labels
		return test_datas, test_labels

