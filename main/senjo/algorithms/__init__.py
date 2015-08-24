#from .grabed import grabed
import cv2
import numpy as np

class OpenCVAlgorithm(object):
	def __init__(self, feature_detector_name, descriptor_extractor_name):
		self.feature_detector_instance = cv2.FeatureDetector_create(feature_detector_name)
		self.descriptor_extractor_instance = cv2.DescriptorExtractor_create(descriptor_extractor_name)

	def compute(self, image_path):
		img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		kp = self.feature_detector_instance.detect(img)
		kp, desc = self.descriptor_extractor_instance.compute(img,kp)
		return np.float32(desc)

class DSIFT(OpenCVAlgorithm):
	def __init__(self):
		super(DSIFT,self).__init__('Dense', 'SIFT')

class SIFT(OpenCVAlgorithm):
	def __init__(self):
		super(SIFT,self).__init__('SIFT', 'SIFT')

class SURF(OpenCVAlgorithm):
	def __init__(self):
		super(SURF,self).__init__('SURF', 'SURF')

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
	def __init__(self, descriptor_extractor, descriptor_matcher):
		self.descriptor_matcher = descriptor_matcher
		self.descriptor_extractor = descriptor_extractor

	def setVocabulary(self, visual_dictionary):
		self.vocabulary = visual_dictionary
		print 'add visual_dictionary', visual_dictionary.shape
		# TODO: Why the hell this doesnt work? trainIdx is wrong if we do that
		#self.descriptor_matcher.add(visual_dictionary)

	def compute(self, img_path):
		desc = self.descriptor_extractor.compute(img_path)
		print desc.shape
		output_descriptors = self._compute(desc)
		return np.float32(output_descriptors)

	def _compute(self, query_descriptor):
		cluster_count = self.vocabulary.shape[0]

		matches = self.descriptor_matcher.match(query_descriptor, self.vocabulary)
		output_descriptor = np.zeros((cluster_count,), dtype=np.float32)

		for i,m in enumerate(matches):
			assert m.queryIdx == int(i)
			output_descriptor[m.trainIdx] += float(1.0)
		# Normalize
		output_descriptor /= query_descriptor.shape[0]
		return output_descriptor
