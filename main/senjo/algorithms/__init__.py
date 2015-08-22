#from .grabed import grabed
import cv2
import numpy as np

class SIFT(object):
	def __init__(self):
		self.feature_detector_instance = cv2.FeatureDetector_create('Dense')
		self.descriptor_extractor_instance = cv2.DescriptorExtractor_create('SIFT')

	def compute(self, image_path):
		img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		kp = self.feature_detector_instance.detect(img)
		kp, desc = self.descriptor_extractor_instance.compute(img,kp)
		desc = np.float32(desc)
		return desc.ravel()

class SVM(object):
	def __init__(self):
		self.svm = cv2.SVM()
		self.svm_params = dict(
			kernel_type = cv2.SVM_RBF,
			svm_type = cv2.SVM_C_SVC,
			C=1,
			gamma=0.5
		)

	def train(self, training_data, labels):
		return self.svm.train(training_data, labels, params=self.svm_params)

	def save(self, filename, name='default_classifier_name'):
		return self.svm.save(filename, name)

	def load(self, filename, name='default_classifier_name'):
		return self.svm.load(filename, name)

	def predict(self, *args, **kwargs):
		return self.svm.predict(*args, **kwargs)

	def predict_all(self, *args, **kwargs):
		return self.svm.predict_all(*args, **kwargs)

	def to_json(self):
		return dict(params=self.svm_params)

from grabed import grabed, default_border_extractor
class GRABED(object):
	def __init__(self, scales=1, f=0.5, max_axis=128, l0=3, r=12, border_extractor=default_border_extractor, output_file=None):
		self.scales = scales
		self.f = f
		self.max_axis = max_axis
		self.l0 = l0
		self.r = r
		self.border_extractor = border_extractor
		self.output_file = output_file

	def compute(self, image_path):
		img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		desc = grabed(img, self.scales, self.f, self.max_axis, self.l0, self.r, self.border_extractor, self.output_file)
		return np.float32(desc)

class BOWImgDescriptorExtractor(object):
	def __init__(self, extractor, matcher, feature_detector_instance=None):
		self.feature_detector_instance = feature_detector_instance or cv2.FeatureDetector_create('Dense')
		self.descriptor_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)

	def setVocabulary(visual_dictionary):
		self.descriptor_extractor.setVocabulary(visual_dictionary)

	def compute(self, img_path):
		img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

		kp = self.feature_detector_instance.detect(img)
		desc = classifier_descriptor_extractor_instance.compute(img, kp)

		return np.float32(desc)
