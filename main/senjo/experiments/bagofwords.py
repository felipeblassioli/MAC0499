import logging
logger = logging.getLogger('senjo')

from senjo.experiments import Experiment
from senjo.algorithms import BOWImgDescriptorExtractor

import numpy as np
import cv2
import os

class BOWExperiment(Experiment):
	"""Uses self.clusterizer to cluster the descriptors computed by self.descriptor_extractor this gives a visual dictionary
	Then uses self.bow_descriptor_extractor(which has setVocabulary) loaded with the visual_dictionary to compute the descriptors that are input to the classifier.
	"""
	cluster_count = 8
	visual_dictionary_file = None

	@property
	def clusterizer(self):
		"""Has cluster(unclusterized_features) method"""
		cluster_count = self.cluster_count
		# BOWKMeansTrainer params

		#Number of bags
		#cluster_count = 100
		# 100 iter or accuracy of 0.001
		term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		attempts = 1
		#flags = cv2.KMEANS_RANDOM_CENTERS
		flags = cv2.KMEANS_PP_CENTERS
		return cv2.BOWKMeansTrainer(cluster_count, term_criteria, attempts, flags)

	@property
	def descriptor_matcher(self):
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50) # or pass empty dictionary
		return cv2.FlannBasedMatcher(index_params, search_params)

	@property
	def bow_descriptor_extractor(self):
		"""Has:
		setVocabulary(visual_dictionary)
		compute(image_path) -> descriptor
		"""
		return BOWImgDescriptorExtractor(self.descriptor_extractor, self.descriptor_matcher)

	def build_visual_dictionary(self, training_dataset, descriptor_extractor_instance, clusterizer_instance, visual_dictionary_file, output_dir):
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
			desc = descriptor_extractor_instance.compute(image_path)
			features_unclustered.extend(desc)

		features_unclustered = np.vstack(features_unclustered)
		visual_dictionary = clusterizer_instance.cluster(features_unclustered)

		logger.debug('Saving visual_dictionary to %s ... ' % visual_dictionary_file)
		np.savetxt(visual_dictionary_file, visual_dictionary)
		logger.info('visual dictionary built successfully.')

		return visual_dictionary

	def prepare(self, *args, **kwargs):
		training_dataset = kwargs['training_dataset']
		output_dir = kwargs['output_dir']
		classifier_file = kwargs['classifier_file']
		classifier_instance = self.classifier
		descriptor_extractor_instance = self.descriptor_extractor
		clusterizer_instance = self.clusterizer
		bow_descriptor_extractor_instance = self.bow_descriptor_extractor
		name = self.name or 'default_name'
		visual_dictionary_file = self.visual_dictionary_file or '%s-dictionary.txt' % self.name

		visual_dictionary = self.build_visual_dictionary(training_dataset, descriptor_extractor_instance, clusterizer_instance, visual_dictionary_file, output_dir)
		bow_descriptor_extractor_instance.setVocabulary(visual_dictionary)
		return self.build_model(training_dataset, classifier_instance, bow_descriptor_extractor_instance, output_dir, classifier_file)
