from senjo.experiments import Dataset

dataset = Dataset('instance/data/botas_e_sapatos')
for img_path, label in dataset:
	print img_path, label


