import logging
logger = logging.getLogger('senjo')

import os
import cv2
import numpy as np

from senjo.helpers import DefaultJSONEncoder
from senjo.cache import Cache

class ExperimentResultWrapper(object):
	def __init__(self, algorithm_name, training_dataset, test_dataset, samples, labels, predictions, extras=None):
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

		def _wrap(f):
			try:
				return f()
			except ZeroDivisionError:
				return 0.0
		for k,v in _samples.items():
			_samples[k]['TN'] = _samples[k]['total'] - sum([ v for v in _samples[k].values() if k in ['TP','FP','FN']])
			TP = float(_samples[k]['TP'])
			FP = float(_samples[k]['FP'])
			TN = float(_samples[k]['TN'])
			FN = float(_samples[k]['FN'])
			_samples[k]['statistics'] = dict(
				#sensitivity or true positive rate (TPR)
				TPR=_wrap(lambda: TP/(TP+FN)),
				# specificity (SPC) or true negative rate (TNR)
				TNR=_wrap(lambda: TN/(FP+TN)),
				# precision or positive predictive value (PPV)
				PPV=_wrap(lambda: TP/(TP+FP)),
				# negative predictive value (NPV)
				NPV=_wrap(lambda: TN/(TN+FN)),
				# fall-out or false positive rate (FPR)
				FPR=_wrap(lambda: FP/(FP+TN)),
				# false discovery rate (FDR)
				FDR=_wrap(lambda: FP/(FP+TP)),
				# miss rate or false negative rate (FNR)
				FNR=_wrap(lambda: FN/(FN+TP)),
				# https://en.wikipedia.org/wiki/Accuracy_and_precision
				ACC=_wrap(lambda: (TP+TN)/(TP+FP+FN+TN)),
				# https://en.wikipedia.org/wiki/F1_score
				F1=_wrap(lambda: 2*TP/(2*TP+FP+FN)),
				# https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
				MCC=_wrap(lambda: ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
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
	def __init__(self, dataset_root_dir, name=None, mapping_file='classes_mapping.txt'):
		import os
		dataset = []
		# directory's names are category labels
		for category in os.listdir(dataset_root_dir):
			category_dir = os.path.join(dataset_root_dir, category)
			if os.path.isdir(category_dir):
				for filename in os.listdir(category_dir):
					file_path = os.path.join(category_dir, filename)
					dataset.append((file_path, [float(category)]))

		mapping_file = os.path.join(dataset_root_dir, mapping_file)
		try:
			with open(mapping_file, 'rb') as fp:
				self.categories = [ tuple([ x.strip() for x in line.split(' ')]) for line in fp ]
		except IOError:
			self.categories = []
		self._data = dataset
		self.name = name or dataset_root_dir
		logging.debug('dataset length: %d' % len(self._data))

	def __iter__(self):
		for x in self._data:
			yield x

	def to_json(self):
		return dict(
			name=self.name,
			categories=self.categories,
			data=self._data
		)

from senjo.cache import Cache
from senjo.algorithms import SVM, DSIFT
import cv2
import numpy as np
import os
class Experiment(object):
	@property
	def classifier(self):
		return SVM()

	@property
	def descriptor_extractor(self):
		return DSIFT()

	def __init__(self, training_dataset=None, test_dataset=None, name=None, classifier_file=None, output_dir=None):
		self.training_dataset = training_dataset
		self.test_dataset = test_dataset
		self.name = name
		self.classifier_file = classifier_file
		self.output_dir = output_dir

	def run(self, training_dataset=None, test_dataset=None, name=None, output_dir=None, classifier_file=None):
		training_dataset = self.load_dataset(training_dataset or self.training_dataset)
		test_dataset = self.load_dataset(test_dataset or self.test_dataset)
		name = name or self.name or 'default_experiment_name'
		output_dir = output_dir or self.output_dir or 'experiments_output'
		classifier_file = classifier_file or self.classifier_file or '%s-classifier.xml' % name

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		logger.info('\nRunning experiment %s ...' % name)

		run_kwargs = locals()
		run_kwargs.pop('self')
		model, tests_descriptor_extractor = self.prepare(**run_kwargs)

		logger.info('testing...')

		samples, labels = self.make_test_samples(tests_descriptor_extractor, test_dataset, name, output_dir)
		predictions = model.predict_all(samples)

		logger.info('testing end.')

		_samples = [ img_path for img_path, _ in test_dataset ]
		result = self.make_result(name, training_dataset, test_dataset, _samples, labels, predictions)

		logger.info('experient end.\n')
		return result

	def prepare(self, *args, **kwargs):
		training_dataset = kwargs['training_dataset']
		output_dir = kwargs['output_dir']
		classifier_file = kwargs['classifier_file']
		classifier_instance = self.classifier
		descriptor_extractor_instance = self.descriptor_extractor
		return self.build_model(training_dataset, classifier_instance, descriptor_extractor_instance, output_dir, classifier_file)

	def build_model(self, training_dataset, classifier_instance, descriptor_extractor_instance, output_dir, classifier_file):
		classifier_file = os.path.join(output_dir, classifier_file)

		if os.path.isfile(classifier_file):
			logger.info('Loading classifier from file %s.' % classifier_file)
			classifier_instance.load(classifier_file)
		else:
			logger.info('Building classifier...')

			training_data = []
			labels = []
			for img_path, img_categories in training_dataset:
				desc = descriptor_extractor_instance.compute(img_path)
				training_data.append(desc)
				labels.append(img_categories[0])

			training_data = np.vstack(training_data)
			labels = np.vstack(np.array(labels, dtype=np.float32))

			classifier_instance.train(training_data, labels)
			logger.info('saving classifier to %s ...' % classifier_file)
			classifier_instance.save(classifier_file)
			logger.info('classifier built successfully.')

		return classifier_instance, descriptor_extractor_instance

	def make_test_samples(self, DE, test_dataset, name=None, cache_dir=None):
		"""Look for cached descriptors or compute them using DE. These descriptors are our samples."""
		_cache = Cache(name=name, root_dir=cache_dir) if cache_dir is not None and name is not None else None
		_samples = []
		for img_path, _ in test_dataset:
			print img_path
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

		samples = np.vstack(_samples)
		labels = np.vstack([ categories[0] for _, categories in test_dataset ])
		return (samples, labels)

	def make_result(self, name, training_dataset, test_dataset, samples, labels, predictions, extras=None):
		return ExperimentResultWrapper(name, training_dataset, test_dataset, samples, labels, predictions, extras)

	def load_dataset(self, dataset_root_dir):
		import six
		if isinstance(dataset_root_dir, six.string_types):
			return Dataset(dataset_root_dir)
		else:
			# assume it is a loaded dataset
			return dataset_root_dir

import bagofwords
BOWExperiment = bagofwords.BOWExperiment
