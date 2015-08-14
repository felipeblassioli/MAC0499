import logging
logger = logging.getLogger('senjo')

# SIFT, SURF, BRIEF, BRISK, ORB

import cv2
import numpy as np
import os
def build_visual_dictionary(training_dataset, feature_detector_instance, descriptor_extractor_instance, visual_dictionary_file, clusterizer_instance, output_dir):
	visual_dictionary_file = os.path.join(output_dir, visual_dictionary_file)
	try:
		visual_dictionary = np.loadtxt(visual_dictionary_file, dtype=np.float32)
		logger.info('Visual dictionary read from %s.' % visual_dictionary_file)
		return visual_dictionary
	except IOError:
		pass

	logger.info('Building visual dictionary..')

	features_unclustered = []

	for image_path, _ in training_dataset:
		logger.debug('extracting feature vector of %s' % image_path)

		img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		kp = feature_detector_instance.detect(img)
		kp, desc = descriptor_extractor_instance.compute(img,kp)
		desc = np.float32(desc)
		features_unclustered.append(desc)

	features_unclustered = np.vstack(desc)
	print features_unclustered.dtype
	visual_dictionary = clusterizer_instance.cluster(features_unclustered)

	np.savetxt(visual_dictionary_file, visual_dictionary)
	logger.info('visual dictionary built successfully.')

	return visual_dictionary

def build_model(training_dataset, visual_dictionary, feature_detector_instance, classifier_descriptor_extractor_instance, classifier_instance, classifier_file, output_dir):
	classifier_file = os.path.join(output_dir, classifier_file)

	classifier_descriptor_extractor_instance.setVocabulary(visual_dictionary)
	class _DescriptorExtractor(object):
		def compute(self, img_path):
			img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

			kp = feature_detector_instance.detect(img)
			desc = classifier_descriptor_extractor_instance.compute(img, kp)

			return np.float32(desc)

	_descriptor_extractor = _DescriptorExtractor()

	if os.path.isfile(classifier_file):
		logger.info('Loading classifier from file %s.' % classifier_file)
		classifier_instance.load(classifier_file)
	else:
		logger.info('Building classifier using visual dictionary..')

		training_data = []
		labels = []
		for img_path, img_categories in training_dataset:

			logger.debug('translating %s to visual words..' % img_path)

			desc = _descriptor_extractor.compute(img_path)
			training_data.append(desc)
			labels.append(img_categories[0])

		training_data = np.vstack(training_data)
		labels = np.vstack(np.array(labels, dtype=np.float32))

		classifier_instance.train(training_data, labels)
		classifier_instance.save(classifier_file)
		logger.info('classifier built successfully.')

	return classifier_instance, _descriptor_extractor

class SVMWrapper(object):
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
