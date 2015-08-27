#from .grabed import grabed
import cv2
import numpy as np

from six import string_types
class BaseDescriptorExtractor(object):
	def compute(self, image_path):
		if isinstance(image_path, string_types):
			img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		else:
			img = image_path
		return self._compute(img)

	def _compute(self, img):
		raise NotImplementedError()

class OpenCVAlgorithm(BaseDescriptorExtractor):
	def __init__(self, feature_detector_name, descriptor_extractor_name):
		self.feature_detector_instance = cv2.FeatureDetector_create(feature_detector_name)
		self.descriptor_extractor_instance = cv2.DescriptorExtractor_create(descriptor_extractor_name)

	def _compute(self, img):
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
class GRABED(BaseDescriptorExtractor):
	def __init__(self, scales=1, f=0.5, max_axis=128, l0=3, r=12, border_extractor=default_border_extractor, output_file=None):
		self.scales = scales
		self.f = f
		self.max_axis = max_axis
		self.l0 = l0
		self.r = r
		self.border_extractor = border_extractor
		self.output_file = output_file

	def _compute(self, img):
		desc = grabed(img, self.scales, self.f, self.max_axis, self.l0, self.r, self.border_extractor, self.output_file)
		return np.float32(desc)

class BOWImgDescriptorExtractor(object):
	def __init__(self, descriptor_extractor, descriptor_matcher):
		self.descriptor_matcher = descriptor_matcher
		self.descriptor_extractor = descriptor_extractor

	def setVocabulary(self, visual_dictionary):
		self.vocabulary = visual_dictionary
		# TODO: Why the hell this doesnt work? trainIdx is wrong if we do that
		#self.descriptor_matcher.add(visual_dictionary)

	def compute(self, img_path):
		desc = self.descriptor_extractor.compute(img_path)
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

import cv2
import numpy as np

from collections import namedtuple
Point = namedtuple("Point", ["x","y"])

class _QEF(object):
	def __init__(self, levels=2, shape=(3,2)):
		self.levels = levels
		self.shape = shape

	def getRects(self, img):
		def pairwise(iterable):
			from collections import deque
			from itertools import izip
			a = iter(iterable)
			b = deque(iterable)
			b.popleft()
			return izip(a,b)

		width, height = img.shape[0], img.shape[1]
		l1 = np.linspace(0, height, self.shape[1]+1)
		l2 = np.linspace(0, width, self.shape[0]+1)
		return [ ( Point(int(x1),int(y1)), Point(int(x2),int(y2)) ) for x1,x2 in pairwise(l1) for y1,y2 in pairwise(l2) ]

	def getROIs(self, img):
		rects = self.getRects(img)
		for p,q in rects:
			yield img[p.y:q.y, p.x:q.x]

	def compute(self, img):
		return self.getRects(img)

class QEF(BaseDescriptorExtractor):
	def __init__(self, descriptor_extractor, levels=2, shape=(3,2)):
		self.q = _QEF(levels, shape)
		self.descriptor_extractor = descriptor_extractor

	def compute(self, image_path):
		img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		return self._compute(img)

	def _compute(self, img):
		descriptors = []
		for roi in self.q.getROIs(img):
			desc = self.descriptor_extractor.compute(roi)
			descriptors.append(desc)
		for d in descriptors:
			print d.shape
		return np.hstack(descriptors)
