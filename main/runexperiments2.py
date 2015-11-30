from senjo.experiments import Experiment, BOWExperiment
from senjo.algorithms import GRABED
class GRABEDExperiment(Experiment):
	@property
	def classifier(self):
		return SVM(C=1, gamma=0.5)
	@property
	def descriptor_extractor(self):
		return GRABED()

class BOWGRABEDExperiment(BOWExperiment):
	@property
	def descriptor_extractor(self):
		return GRABED()

res = GRABEDExperiment('instance/data/data02', 'instance/data/data02-tr', name='GRABED').run()
res.save('result-grabed.json')

res = BOWGRABEDExperiment('instance/data/data02', 'instance/data/data02-tr', name='BOW-GRABED').run()
res.save('result-bow-grabed.json')


