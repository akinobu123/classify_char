import tensorflow as tf
import sys
#from supply_mnist import Supply
from supply_charimg import Supply
from model_classifychar import Model
from util import Util
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont


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

		# start train NN
		for i in range (20000):
			print('.', end='')
			sys.stdout.flush()
		
			# train the NN model.
			train_images, train_labels = supply.next_train_batch (100)
			sess.run (train, feed_dict = {
				images: train_images, 
				labels: train_labels, keep: 0.5})
			# compute the accuracy of the trained NN model every 100 times.
			if i % 10 == 0:
				print ('')
				print ('step %d' % (i))
				test_images, test_labels = supply.get_test_data ()
				out_labels = sess.run (infer, feed_dict = {
					images: test_images, 
					labels: test_labels, keep: 1.0})

#				for out_label, label, image in zip(out_labels, test_labels, test_images):
#					index_out_label = numpy.argmax(numpy.array(out_label), 0)
#					index_label = numpy.argmax(numpy.array(label), 0)
#					if index_out_label != index_label:
#						image = image.reshape (64, 64)
#						pil_image = Image.fromarray(image, "L")
#						pil_image.save(str(supply.label_to_sjis(numpy.argmax(label))) + '.png', 'PNG')

#						cv2.destroyAllWindows()
#						cv2.imshow('image', image)
#						cv2.waitKey(0)
#						print ('')
#						print ('out-label %s, label %s' % (index_out_label, index_label))

				test_images, test_labels = supply.get_test_data ()
				test_accuracy = sess.run (accuracy, feed_dict = {
					images: test_images, 
					labels: test_labels, keep: 1.0})
				print ('')
				print ('step %d, training accuracy %.2f' % (i, test_accuracy))

		cv2.destroyAllWindows()
		saver.save(sess, "./model_save/model.ckpt")

