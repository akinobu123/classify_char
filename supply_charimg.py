import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy
import random
from data_aug import DataAug


class Supply:

	def get_output_dim (self):
		# returns output dimension (label dimension)
		return 11519

	def next_train_batch (self, batch_size):
		# make character list
		sjis_list, char_list = self.create_sjis_list ()

		# make label and image
		train_images = []
		train_labels = []
		for i in range (batch_size):
			# randomize character-code
			index = random.randrange(len(sjis_list))
			sjis = sjis_list [index]
			char = char_list [index]
			# convert to label (one-hot)
			sjis_int = int.from_bytes(sjis, 'big')
			label = self.sjis_to_label (sjis_int)
			label_onehot = self.label_to_onehot (self.get_output_dim(), label)
			# create character image
			image = self.create_char_img (char)
			# data augmentation
			image = self.do_data_aug (image)
			# append data
			train_images.append (image)
			train_labels.append (label_onehot)
		return train_images, train_labels

	def get_test_data (self):
		test_datas, test_labels = self.next_train_batch (100)
		return test_datas, test_labels

	### private functions ###

	def create_char_img (self, char):
		font_name = random.choice ([
			'fonts-japanese-mincho.ttf', 
			'fonts-japanese-gothic.ttf', 
			'TakaoGothic.ttf', 
			'TakaoExGothic.ttf', 
			'TakaoPGothic.ttf', 
			'TakaoMincho.ttf', 
			'TakaoExMincho.ttf', 
			'TakaoPMincho.ttf', 
		])
		back_color = 255 - round(abs(numpy.random.randn() * 20))
		font_color = 0 + round(abs(numpy.random.randn() * 50))
		font_size = 48
		x_offset = 0
		y_offset = 0
		x = (64 - font_size) // 2 + x_offset
		y = (64 - font_size) // 2 + y_offset
		font = ImageFont.truetype(font_name, font_size, encoding='unic')
		image = Image.new('L', (64, 64), back_color)
		draw = ImageDraw.Draw(image)
		draw.text((x, y), char, font = font, fill = font_color)
		image_array = numpy.asarray (image)
		image_array = image_array.reshape (64, 64, 1)
		return image_array

	def do_data_aug (self, image):
		data_aug = DataAug ()
		image = data_aug.scale_rotate_shift (image, 10, 10, 4, 4, 4)
		image = data_aug.rescale (image, 0.1)
		image = data_aug.invert (image, 0.1)
		image = data_aug.noise (image, 0.7)
		image = data_aug.binarize (image, 0.3)
		return image

	def create_sjis_list (self):
		sjis_list = []
		char_list = []
		sjis_range_list = [
#			[b'\x81\x40', b'\x81\xff'],	# 記号     [    0 -   255] 1
			[b'\x82\x40', b'\x83\x96'],	# 英数かな [  256 -   768] 2,3
			[b'\x88\x90', b'\x9f\xff'],	# 漢字     [ 1792 -  7935] 8-
			[b'\xe0\x40', b'\xea\xaf'],	# 漢字     [ 7936 - 10751]
			[b'\xfa\x5c', b'\xfc\x4f'],	# 漢字     [10752 - 11519]
		]
#		sjis_range_list = [
#			[b'\x82\x40', b'\x83\x96'],	# 英数かな [  256 -   768] 2,3
#			[b'\x88\x00', b'\x90\xff'], # - 4096
#		]
		for sjis_range in sjis_range_list:
			start = int.from_bytes(sjis_range[0], 'big')
			end   = int.from_bytes(sjis_range[1], 'big')
			for sjis in range (start, end+1):
				sjis_byte = sjis.to_bytes(2, 'big')
				try:
					char = sjis_byte.decode('sjis', 'strict')
					sjis_list.append (sjis_byte)
					char_list.append (char)
				except:
					pass
		return sjis_list, char_list

	def sjis_to_label (self, sjis):
		'''
		sjis(16)	sjis(10)	label(10)
		-----------------------------
		8100		33024		0
		|			| (7936)	| (sjis - 33024)
		9FFF		40959		7935

		E000		57344		7936
		|			| (2816)	| (sjis - 49408)
		EAFF		60159		10751

		FA00		64000		10752
		|			| (768)		| (sjis - 53248)
		FCFF		64767		11519
		'''
		if sjis >= 33024 and sjis <= 40959:
			return sjis - 33024
		if sjis >= 57344 and sjis <= 60159:
			return sjis - 49408
		if sjis >= 64000 and sjis <= 64767:
			return sjis - 53248

	def label_to_sjis (self, label):
		if label >=     0 and label <=  7935:
			return label + 33024
		if label >=  7936 and label <= 10751:
			return label + 49408
		if label >= 10752 and label <= 11519:
			return label + 53248

	def label_to_onehot (self, max_value, value):
		onehot = numpy.zeros (max_value)
		onehot [value] = 1.0
		return onehot

	def onehot_to_label (self, onehot):
		return numpy.argmax (onehot)


if __name__ == '__main__':
	data_sup = Supply ()
	train_images, train_labels = data_sup.next_train_batch (10)
	#print (train_images.shape())
	for image, label in zip(train_images, train_labels):
		image = image.reshape (64, 64)
		pil_image = Image.fromarray(image, "L")
		pil_image.save(str(data_sup.label_to_sjis(numpy.argmax(label))) + '.png', 'PNG')

