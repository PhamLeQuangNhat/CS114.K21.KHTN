# import the necessary packages
from skimage import feature
import numpy as np
class HOG:
	def __init__(self, orientations):
		# store the orientations
		self.orientations = orientations
	def describe(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hg = feature.hog(gray, orientations=self.orientations, pixels_per_cell=(10, 10),
			cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
		# return the histogram of Local Binary Patterns
		return hg
