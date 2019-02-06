import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
import numpy
import random
import itertools


class DataAug:

	# specify max value
	# scale ... +-parcentage (int)
	# rot   ... degree
	# shift ... pixel
	def scale_rotate_shift (self, in_image, scale_w_max, scale_h_max, 
							rot_max, shift_x_max, shift_y_max, backcolor=0):
		# in_image = [height, width, channels]

		# decide random values
		scale_w = round (numpy.random.randn() * scale_w_max)	# +-parcentage
		scale_h = round (numpy.random.randn() * scale_h_max)	# +-parcentage
		rot = round (numpy.random.randn() * rot_max)			# degree
		shift_x = round (numpy.random.randn() * shift_x_max)	# pixel
		shift_y = round (numpy.random.randn() * shift_y_max)	# pixel

		# scale
		in_h, in_w = in_image.shape[:2]
		scale_w_f = 1.0 + (scale_w / 100)
		scale_h_f = 1.0 + (scale_h / 100)
		dst_w = round (in_w * scale_w_f)
		dst_h = round (in_h * scale_h_f)
		scaled_image = cv2.resize (in_image, (dst_w, dst_h), interpolation = cv2.INTER_CUBIC)

		# rotate
		scaled_h, scaled_w = scaled_image.shape[:2]
		center_x = int (scaled_w / 2)
		center_y = int (scaled_h / 2)
		affine_mat = cv2.getRotationMatrix2D ((center_x, center_y), rot,  1.0)

		# shift
		affine_mat[0][2] += shift_x
		affine_mat[1][2] += shift_y

		# convert image
		out_image = cv2.warpAffine (scaled_image, affine_mat, (in_h, in_w), 
					flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		return out_image.reshape (in_h, in_w, 1)

	def is_probability (self, probability):
		if probability > random.random():	# [0.0, 1.0)
			return True
		else:
			return False

	def rescale (self, in_image, probability):
		if self.is_probability (probability):
			in_h, in_w = in_image.shape[:2]
			in_image = cv2.resize (in_image, (in_w//2, in_h//2), 
				interpolation = cv2.INTER_NEAREST)
			in_image = cv2.resize (in_image, (in_w, in_h), 
				interpolation = cv2.INTER_NEAREST)
			return in_image.reshape (in_h, in_w, 1)
		else:
			return in_image

	def invert (self, in_image, probability):
		if self.is_probability (probability):
			in_h, in_w = in_image.shape[:2]
			in_image = cv2.bitwise_not (in_image)
			return in_image.reshape (in_h, in_w, 1)
		else:
			return in_image

	def noise (self, in_image, probability, white=0.03, black=0.03):
		# in_image = [height, width, channels]
		if self.is_probability (probability):
			in_h, in_w = in_image.shape[:2]
			for x, y in itertools.product(*map(range, (in_w, in_h))):
				r = random.random()
				if r < white:
					in_image[x,y,0] = 255	# white
				elif r > 1 - black:
					in_image[x,y,0] = 0		# black
			return in_image.reshape (in_h, in_w, 1)
		else:
			return in_image

	def binarize (self, in_image, probability):
		if self.is_probability (probability):
			in_h, in_w = in_image.shape[:2]
			_, in_image = cv2.threshold(in_image, 125, 255, cv2.THRESH_BINARY)
			return in_image.reshape (in_h, in_w, 1)
		else:
			return in_image
		
		

