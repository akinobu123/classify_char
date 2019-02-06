import tensorflow as tf
import sys
#from supply_mnist import Supply
from supply_charimg import Supply
from model_classifychar import Model


if __name__ == '__main__':
	# make train data supplier
	supply = Supply ()
	output_dim = supply.get_output_dim ()
	print ('output-dim : %d' % (output_dim))

	with tf.Session() as sess:

		# make NN model
		model = Model ()
		infer, train, accuracy, images, labels, keep = model.make_models (output_dim)

		# initialize NN model
		sess.run (tf.global_variables_initializer())

		# start train NN
		for i in range (40000):
			print('.', end='')
			sys.stdout.flush()
		
			# train the NN model.
			train_images, train_labels = supply.next_train_batch (100)
			sess.run (train, feed_dict = {
				images: train_images, 
				labels: train_labels, keep: 0.5})
			# compute the accuracy of the trained NN model every 100 times.
			if i % 10 == 0:
				test_images, test_labels = supply.get_test_data ()
				test_accuracy = sess.run (accuracy, feed_dict = {
					images: test_images, 
					labels: test_labels, keep: 1.0})
				print ('')
				print ('step %d, training accuracy %.2f' % (i, test_accuracy))

		saver = tf.train.Saver()
		saver.save(sess, "./model_save/model.ckpt")

