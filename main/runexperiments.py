from senjo.experiments import Experiment, BOWExperiment

from senjo.algorithms import GRABED, SIFT
class GRABEDExperiment(Experiment):
	@property
	def descriptor_extractor(self):
		return GRABED()

import cv2
class SIFTBOWExperiment(BOWExperiment):
	@property
	def feature_detector(self):
		return cv2.FeatureDetector_create('Dense')

	@property
	def descriptor_extractor(self):
		return cv2.DescriptorExtractor_create('SIFT')
		#return SIFT()

EXPERIMENTS = [
	#Experiment('instance/data/food', 'instance/data/food-tr')
	GRABEDExperiment('instance/data/data01', 'instance/data/data01-tr'),
	GRABEDExperiment('instance/data/data02', 'instance/data/data02-tr'),
	SIFTBOWExperiment('instance/data/data02', 'instance/data/data02-tr'),
]

import logging
logger = logging.getLogger('senjo')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

import os
experiments_dir = 'instance/experiments'
for i,exp in enumerate(EXPERIMENTS):
	if not hasattr(exp, 'name'):
		experiment_name = 'experiment-%03d' % i
	else:
		experiment_name = exp.name
	output_dir = os.path.join(experiments_dir, experiment_name)

	res = exp.run(name=experiment_name, output_dir=output_dir)

	filename = 'result-%s.json' % experiment_name
	result_filepath = os.path.join(experiments_dir, filename)
	res.save(result_filepath)
