import logging
logger = logging.getLogger('senjo')

import numpy as np
import cv2
import os

from stuff import *
from .helpers import DefaultJSONEncoder
from .cache import Cache

class ExperimentResultWrapper(object):
	def __init__(self, training_dataset, test_dataset, algorithm_name, samples, labels, predictions, extras=None):
		self.training_dataset = training_dataset
		self.test_dataset = test_dataset
		self.algorithm_name = algorithm_name
		self.samples = samples
		self.labels = labels
		self.predictions = predictions
		self.extras = extras

		err = (labels != predictions).mean()
		print 'error: %.2f %%' % (err*100)
		self.error = err

		n = len(test_dataset.categories)
		print test_dataset.categories
		confusion = np.zeros((n,n), np.int32)
		for i, j in zip(labels, predictions):
			i,j = int(i)-1, int(j)-1
			confusion[i,j] += 1
		self.confusion_matrix = confusion
		print 'confusion matrix:'
		print confusion
		print
		#print np.mean(samples
	def _group_samples_by_category(self):
		import math
		from collections import defaultdict
		categories = self.training_dataset.categories
		def _bla():
			res = dict()
			res['total'] = 0
			res['TP'] = 0
			res['TN'] = 0
			res['FP'] = 0
			res['FN'] = 0
			res['data'] = []
			return res
		_samples = defaultdict(_bla)
		for img_path, label, prediction in zip(self.samples, self.labels.ravel(), self.predictions.ravel()):
			label, prediction = str(label), str(prediction)
			row = (img_path, prediction)
			_samples[label]['total'] += 1
			if label != prediction:
				_samples[prediction]['FP'] += 1
				_samples[label]['FN'] += 1
			else:
				_samples[label]['TP'] += 1
			_samples[label]['data'].append(row)

		for k,v in _samples.items():
			_samples[k]['TN'] = _samples[k]['total'] - sum([ v for v in _samples[k].values() if k in ['TP','FP','FN']])
			TP = float(_samples[k]['TP'])
			FP = float(_samples[k]['FP'])
			TN = float(_samples[k]['TN'])
			FN = float(_samples[k]['FN'])
			_samples[k]['statistics'] = dict(
				#sensitivity or true positive rate (TPR)
				TPR=TP/(TP+FN),
				# specificity (SPC) or true negative rate (TNR)
				TNR=TN/(FP+TN),
				# precision or positive predictive value (PPV)
				PPV=TP/(TP+FP),
				# negative predictive value (NPV)
				NPV=TN/(TN+FN),
				# fall-out or false positive rate (FPR)
				FPR=FP/(FP+TN),
				# false discovery rate (FDR)
				FDR=FP/(FP+TP),
				# miss rate or false negative rate (FNR)
				FNR=FN/(FN+TP),
				# https://en.wikipedia.org/wiki/Accuracy_and_precision
				ACC=(TP+TN)/(TP+FP+FN+TN),
				# https://en.wikipedia.org/wiki/F1_score
				F1=2*TP/(2*TP+FP+FN),
				# https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
				MCC=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
			)
		return _samples

	def to_json(self):
		return dict(
			algorithm_name=self.algorithm_name,
			training_dataset=self.training_dataset,
			test_dataset=self.test_dataset,
			samples=self._group_samples_by_category(),
			error=self.error,
			confusion_matrix=self.confusion_matrix,
			extras=self.extras
		)

	@classmethod
	def _validate_extension(cls, filename):
		SUPPORTED_EXTENSIONS = ['.json']
		import os
		filename, ext = os.path.splitext(filename)
		if ext not in SUPPORTED_EXTENSIONS:
			raise Exception('Unsuported extension %s.' % ext)

	@classmethod
	def from_file(cls, filename):
		self._validate_extension(filename)

		with open(filename, 'rb') as fp:
			pass

	def save(self, filename):
		self._validate_extension(filename)

		logger.info('saving result to %s' % filename)
		with open(filename, 'wb') as fp:
			import json
			json.dump(self, fp, cls=DefaultJSONEncoder)

class Dataset(object):
	def __init__(self, dataset_root_dir, name=None):
		import os
		dataset = []
		# directory's names are category labels
		for category in os.listdir(dataset_root_dir):
			category_dir = os.path.join(dataset_root_dir, category)
			if os.path.isdir(category_dir):
				for filename in os.listdir(category_dir):
					file_path = os.path.join(category_dir, filename)
					dataset.append((file_path, [float(category)]))

		mapping_file = os.path.join(dataset_root_dir, 'classes_mapping.txt')
		with open(mapping_file, 'rb') as fp:
			self.categories = [ tuple([ x.strip() for x in line.split(' ')]) for line in fp ]

		self._data = dataset
		self.name = name or dataset_root_dir

	def __iter__(self):
		for x in self._data:
			yield x

	def to_json(self):
		return dict(
			name=self.name,
			categories=self.categories,
			data=self._data
		)

class Experiment(object):
	def load_dataset(self, dataset_root_dir):
		return Dataset(dataset_root_dir)

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
		visual_dictionary = build_visual_dictionary(training_dataset, feature_detector_instance,
													descriptor_extractor_instance, visual_dictionary_file, clusterizer_instance,
													output_dir)
		(model, DE) = build_model(training_dataset, visual_dictionary, feature_detector_instance,
									bow_descriptor_extractor_instance,
									classifier_instance, classifier_file,
									output_dir)
		return model, DE

	def _get_test_samples(self, DE, test_dataset, name=None, cache_dir=None):
		_cache = Cache(name=name, root_dir=cache_dir) if cache_dir is not None and name is not None else None
		_samples = []
		for img_path, _ in test_dataset:
			if _cache is not None:
				if _cache['descriptors'][img_path] is None:
					logger.debug('extracting descriptors for %s' % img_path)
					_cache['descriptors'][img_path] = DE.compute(img_path)
				else:
					logger.debug('using cached descriptor for %s' % img_path)
				_samples.append(_cache['descriptors'][img_path])
			else:
				_samples.append(DE.compute(img_path))

		if _cache is not None:
			_cache.save()
		#samples = np.vstack([ DE.compute(img_path) for img_path, _ in test_dataset ])
		samples = np.vstack(_samples)
		labels = np.vstack([ categories[0] for _, categories in test_dataset ])
		return (samples, labels)

	def run(self, name=None, output_dir=None):
		logger.info('\nRunning experiment %s ...' % name)

		self.params['output_dir'] = output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		model, DE = self._prepare(**self.params)

		logger.info('testing..')

		samples, labels = self._get_test_samples(DE, self.params['test_dataset'], name=name, cache_dir=output_dir)
		predictions = model.predict_all(samples)

		logger.info('testing end.')

		#_samples = zip(self.params['test_dataset'], samples)
		#print _samples
		result = self.make_result([ img_path for img_path, _ in self.params['test_dataset'] ], labels, predictions)

		logger.info('experiment end.\n')
		return result

	def make_result(self, samples, labels, predictions):
		pass

class VanillaExperiment(Experiment):
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
