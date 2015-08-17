import logging
logger = logging.getLogger('senjo')
# create console handler and set level to debug
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#logger.addHandler(ch)
logger.setLevel(logging.INFO)

from senjo.experiment import *

def main():
	training_dataset = 'data/data01'
	test_dataset = 'data/data01-tr'
	ALGORITHMS = ['SIFT', 'SURF' ]
	CLUSTER_COUNTS = [2,4,8,10,16,32,50,100,120]

	def gen_experiments():
		i = 0
		for cc in CLUSTER_COUNTS:
			for an in ALGORITHMS:
				args = [ training_dataset, test_dataset ]
				kwargs = dict(
					algorithm_name = an,
					cluster_count=cc,
					visual_dictionary_filename='cc-vocabulary-%d' % i,
					svm_filename='cc-svm-%d' % i
				)
				print 'FEATURE_DETECTOR', an
				print 'CLUSTER_COUNT', cc
				yield experiment_vanilla(*args, **kwargs)
				i += 1

	experiments = gen_experiments()
	for params in experiments:
		run(**params)
		print '--------------------------------'



EXPERIMENTS = [
	VanillaExperiment(
		'instance/data/data02',
		'instance/data/data02-tr',
		algorithm_name='SURF',
		cluster_count=8,
		visual_dictionary_filename='surf2',
		svm_filename='surf2'
	)
]

import os
experiments_dir = 'instance/experiments'
for i,exp in enumerate(EXPERIMENTS):
	if not hasattr(exp, 'name'):
		experiment_name = 'experiment-%03d' % i
	else:
		experiment_name = exp.name
	output_dir = os.path.join(experiments_dir, experiment_name)

	res = exp.run(name=experiment_name, output_dir=output_dir)

	filename = '%s.json' % experiment_name
	result_filepath = os.path.join(experiments_dir, filename)
	res.save(result_filepath)
