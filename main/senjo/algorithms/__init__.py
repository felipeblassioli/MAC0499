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
