from senjo.experiment import BaseExperiment, ExperimentResultWrapper

import numpy as np
import cv2

class BOWAlgorithm(object):
	@property
	def clusterizer(self):
		cluster_count = self.cluster_count
		# BOWKMeansTrainer params

		#Number of bags
		#cluster_count = 100
		# 100 iter or accuracy of 0.001
		term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		attempts = 1
		#flags = cv2.KMEANS_RANDOM_CENTERS
		flags = cv2.KMEANS_PP_CENTERS
		bow_trainer = cv2.BOWKMeansTrainer(cluster_count, term_criteria, attempts, flags)

		return bow_trainer

	@property
	def bow_descriptor_extractor(self):
		algorithm_name = self.algorithm_name

		# BOWImgDescriptorExtractor params
		extractor = cv2.DescriptorExtractor_create(algorithm_name)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		matcher =  cv2.FlannBasedMatcher(index_params, search_params)

		bow_de = cv2.BOWImgDescriptorExtractor(extractor, matcher)

		return bow_de

	@property
	def classifier(self):
		return SVMWrapper()

class BOWExperiment(BaseExperiment, BOWAlgorithm):
	"""
	This class expects the following properties:
	- bow_descriptor_extractor
	- clusterizer
	- classifier
	"""

	def __init__(self, training_dataset_root, test_dataset_root,
				algorithm_name='SIFT', cluster_count=100,
				visual_dictionary_filename='vocabulary', svm_filename='svm'):

		self.algorithm_name = algorithm_name
		self.cluster_count = cluster_count

		self.params = dict(
			training_dataset = self.load_dataset(training_dataset_root),
			test_dataset = self.load_dataset(test_dataset_root),

			feature_detector_instance = cv2.FeatureDetector_create(algorithm_name),
			descriptor_extractor_instance = cv2.DescriptorExtractor_create(algorithm_name),
			visual_dictionary_file = visual_dictionary_filename + '.txt',

			clusterizer_instance = self.clusterizer,

			bow_descriptor_extractor_instance = self.bow_descriptor_extractor,
			classifier_instance = self.classifier,
			classifier_file = svm_filename + '.xml'
		)

	def _prepare(self,
		training_dataset=None,
		test_dataset=None,
		feature_detector_instance=None,
		descriptor_extractor_instance=None,
		visual_dictionary_file=None,
		clusterizer_instance=None,
		bow_descriptor_extractor_instance=None,
		classifier_instance=None,
		classifier_file=None,
		output_dir=None
	):
	"""Builds a visual dictionary and a model based on the visual dictionary"""
	visual_dictionary = self.build_visual_dictionary(training_dataset, feature_detector_instance,
												descriptor_extractor_instance, visual_dictionary_file, clusterizer_instance,
												output_dir)
	(model, DE) = self.build_model(training_dataset, visual_dictionary, feature_detector_instance,
								bow_descriptor_extractor_instance,
								classifier_instance, classifier_file,
								output_dir)
	return model, DE

	def make_result(self, samples, labels, predictions):
		result = ExperimentResultWrapper(
			self.params['training_dataset'],
			self.params['test_dataset'],
			self.algorithm_name,
			samples,
			labels,
			predictions,
			extras=self.params
		)
		return result

	def build_visual_dictionary(self, training_dataset, feature_detector_instance, descriptor_extractor_instance, visual_dictionary_file, clusterizer_instance, output_dir):
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

	def build_model(self, training_dataset, visual_dictionary, feature_detector_instance, classifier_descriptor_extractor_instance, classifier_instance, classifier_file, output_dir):
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
