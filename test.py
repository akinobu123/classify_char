import tensorflow as tf
import sys
#from supply_mnist import Supply
from supply_charimg import Supply
from model_classifychar import Model


if __name__ == '__main__':
	# make train data supplier
	supply = Supply ()
	output_dim = supply.get_output_dim ()

	with tf.Session() as sess:

		# make NN model
		model = Model ()
		infer, train, accuracy, images, labels, keep = model.make_models (output_dim)

		# load train data
		saver = tf.train.Saver()
		saver.restore(sess, "./model_save/model.ckpt")

		# test
		test_images, test_labels = supply.get_test_data ()
		test_accuracy = sess.run (accuracy, feed_dict = {
			images: test_images, 
			labels: test_labels, keep: 1.0})
		print ('accuracy %.2f' % (test_accuracy))


