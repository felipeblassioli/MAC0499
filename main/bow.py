import cv2
import numpy as np

import os
import random
IMAGES_DIR = 'data/mixed'
images = random.sample([ filename for filename in os.listdir(IMAGES_DIR) ], 20)

features_unclustered = []
extractor = cv2.SIFT()
for filename in images:
	print filename
	filepath = os.path.join(IMAGES_DIR, filename)
	img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	kp = extractor.detect(img)
	kp,desc = extractor.compute(img,kp)

	features_unclustered.append(desc)
	#print desc
features_unclustered = np.vstack(desc)
#Number of bags
cluster_count = 100
# 100 iter or accuracy of 0.001
term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
attempts = 1
#flags = cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_PP_CENTERS
bow_trainer = cv2.BOWKMeansTrainer(cluster_count, term_criteria, attempts, flags)

dictionary = bow_trainer.cluster(features_unclustered)
print 'dictionary'
for x in dictionary:
	print x
print len(dictionary)
#fs = cv2.FileStorage('dictionary.yml', cv2.FILE_STORAGE_WRITE)
print 'dictionary is', type(dictionary)

extractor = cv2.DescriptorExtractor_create('SIFT')

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
matcher =  cv2.FlannBasedMatcher(index_params, search_params)

bow_de = cv2.BOWImgDescriptorExtractor(extractor, matcher)
bow_de.setVocabulary(dictionary)

filename = 'data/nike.jpg'
img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE);
detector = cv2.FeatureDetector_create('SIFT')
kp = detector.detect(img)
bow_descriptor = bow_de.compute(img, kp)

print bow_descriptor
svm_params = dict(
	kernel_type = cv2.SVM_LINEAR,
	svm_type = cv2.SVM_C_SVC,
	C=2.67, gamma=5.383
)

# SVM
svm = cv2.SVM()
svm.train(trainData, responses, params=svm_params)

#mask = result==responses
#correct = np.count_nonzero(mask)
#print correct*100.0/result.size
