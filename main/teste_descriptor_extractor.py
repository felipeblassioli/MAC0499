import cv2
import numpy as np

class OpenCVSIFT(object):
	def compute(self, image_path):
		img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		kp = cv2.FeatureDetector_create('SIFT').detect(img)
		kp, desc = cv2.DescriptorExtractor_create('SIFT').compute(img,kp)

		return np.float32(desc)

descriptor_extractor = OpenCVSIFT()

desc = descriptor_extractor.compute('bota.jpg')
print desc.shape, desc.dtype, type(desc)
print desc
