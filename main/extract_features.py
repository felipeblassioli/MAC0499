# -*- coding: utf-8 -*-
import cv2
import numpy as np

feature_detectors = [
	"FAST",
	"STAR",
	"SIFT",
	"SURF",
	"ORB",
	"BRISK",
	"MSER",
	"GFTT",
	"HARRIS",
	"Dense",
	"SimpleBlob"
]

def identity(img): return img
def grayscale(img): return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def global_threshold(img): return cv2.threshold(grayscale(img),127,255,cv2.THRESH_BINARY)[1]
def otsu_threshold(img): return cv2.threshold(grayscale(img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
def otsu_fgauss_threshold(img): 
	blur = cv2.GaussianBlur(grayscale(img),(5,5),0)
	return  cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def adaptativeThreshold(img): 
	return cv2.adaptiveThreshold(
		grayscale(img),
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		3,
		3
	)

def canny(img):
	thresh = 100;
	return cv2.Canny( grayscale(img), thresh, thresh*2, 3 );

def contours1(img):
	contours, hierarchy = cv2.findContours( adaptativeThreshold(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img,contours,-1,(255,0,0))
	return img

def contours2(img):
	h,w = img.shape[:2]
	h, w = img.shape[:2]
	contours0, hierarchy = cv2.findContours( adaptativeThreshold(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
	def update(levels):
		vis = np.zeros((h, w, 3), np.uint8)
		levels = levels - 3
		cv2.drawContours( vis, contours, (-1, 3)[levels <= 0], (128,255,255),
		3, cv2.CV_AA, hierarchy, abs(levels) )
		return vis
	return update(1)
	#return img

def rgb_histogram(img):

	bins = np.arange(256).reshape(256,1)

	def hist_curve(im):
		h = np.zeros((300,256,3))
		if len(im.shape) == 2:
			color = [(255,255,255)]
		elif im.shape[2] == 3:
			color = [ (255,0,0),(0,255,0),(0,0,255) ]
		for ch, col in enumerate(color):
			hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
			cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
			hist=np.int32(np.around(hist_item))
			pts = np.int32(np.column_stack((bins,hist)))
			cv2.polylines(h,[pts],False,col)
		y=np.flipud(h)
		return y

	# def hist_lines(im):
	# 	h = np.zeros((300,256,3))
	# 	if len(im.shape)!=2:
	# 		print "hist_lines applicable only for grayscale images"
	# 		#print "so converting image to grayscale for representation"
	# 		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# 	hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
	# 	cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
	# 	hist=np.int32(np.around(hist_item))
	# 	for x,y in enumerate(hist):
	# 		cv2.line(h,(x,0),(x,y),(255,255,255))
	# 	y = np.flipud(h)
	# 	return y

	return hist_curve(img)

filters = [ identity, grayscale, canny, adaptativeThreshold ]
threshold_filters = [global_threshold, otsu_threshold, otsu_fgauss_threshold]
#filters += threshold_filters
def drawKeypoints(detector, img):
	kp = detector.detect(img,None)
	img=cv2.drawKeypoints(img,kp)
	return img

# dense=cv2.FeatureDetector_create("Dense")
# kp=dense.detect(imgGray)
# kp,des=sift.compute(imgGray,kp)

def write_label(text, img = None, orientation=0, shape=None, fontScale=1):
	fontFace = cv2.FONT_HERSHEY_SIMPLEX
	color = (0,0,0)
	thickness = 2
	
	if img is None:
		if shape is None:
			width, height = 200,200
		else:
			width, height = shape
		blank_image = np.zeros((height,width,3), np.uint8)
		blank_image[:] = 255
	else:
		width, height = img.shape[1], 80
		blank_image = np.zeros((height, width, 3), np.uint8)
		blank_image[:] = 255
	
	def _write(_img):
		textSize, baseline = cv2.getTextSize( text, fontFace, fontScale, thickness)
		pos = ((_img.shape[1] - textSize[0])/2 , (_img.shape[0] + textSize[1])/2)
		cv2.putText(_img, text, pos, fontFace, fontScale, color, thickness)

	_write(blank_image)
	if img is None:
		if orientation == 1:
			blank_image = np.rot90(blank_image)
		return blank_image
	return np.vstack([blank_image,img])


def extract_features(filename, out="output.jpg"):
	original_img = cv2.imread(filename)

	imgs = []
	filter_labels = []
	_make_labels = True
	for detector_name in feature_detectors:
		_imgs = []
		for f in filters:
			if _make_labels:
				filter_labels.append(write_label(f.__name__, orientation=1, shape=(original_img.shape[1],80), fontScale=1))
			img = f(original_img.copy())
			_imgs.append(drawKeypoints(cv2.FeatureDetector_create(detector_name),img))
		_make_labels = False
		_imgs = np.vstack(_imgs)
		_imgs = write_label(detector_name, _imgs)
		imgs.append(_imgs)
	
	filter_labels = np.vstack(filter_labels)
	empty = np.zeros((80, filter_labels.shape[1], 3), np.uint8)
	empty[:] = 255
	filter_labels = np.vstack([empty, filter_labels])

	output = np.hstack([filter_labels]+imgs)
	cv2.namedWindow("Features Keypoints")
	cv2.imshow("Features Keypoints", output)
	cv2.imwrite(out, output)
	cv2.waitKey(0);

if __name__ == '__main__':
	import sys
	filename = sys.argv[1]
	if len(sys.argv) == 3:
		extract_features(filename, sys.argv[2])
	else:
		extract_features(filename)