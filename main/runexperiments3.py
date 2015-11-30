# -*- encoding: utf-8 -*-
from senjo.experiments import Experiment
from senjo.algorithms import GRABED, SVM

import numpy as np
def experiments_generator(training_dataset, test_dataset):
	# The C parameter trades off misclassification of training examples against simplicity of the decision surface.
	# A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.

	# Intuitively, the gamma parameter defines how far the influence of a single training example reaches, 
	# with low values meaning ‘far’ and 
	# high values meaning ‘close’. 
	# The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.

	# list of tuples ( C, gamma )
	gamma_range = np.logspace(-9,3,13)
	C_range = np.logspace(-2,10,13)

	for C, gamma in zip( C_range, gamma_range )+[(1,0.5)]:
		class _Experiment(Experiment):
			@property
			def classifier(self):
				return SVM(C=C, gamma=gamma)
			@property
			def descriptor_extractor(self):
				return GRABED()
		print 'SVM with parameters', C, gamma
		yield _Experiment(training_dataset, test_dataset)

# import logging
# logger = logging.getLogger('senjo')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# logger.addHandler(ch)

import os
EXPERIMENTS_DIR = 'instance/experiments'
EXPERIMENTS = experiments_generator('instance/data/data02', 'instance/data/data02-tr')
for i, exp in enumerate( EXPERIMENTS ):
	experiment_name = 'SVM-%03d' % i

	output_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
	res = exp.run(name=experiment_name, output_dir=output_dir)

	filename = 'result-%s.json' % experiment_name
	result_filepath = os.path.join(EXPERIMENTS_DIR, filename)
	res.save(result_filepath)

