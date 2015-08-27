import cv2
import numpy as np

# http://stackoverflow.com/questions/15424852/region-of-interest-opencv-python
from collections import namedtuple
Point = namedtuple("Point", ["x","y"])

class QEF(object):
	def __init__(self, levels=2, shape=(3,2)):
		self.levels = levels
		self.shape = shape

	def getRects(self, img):
		def pairwise(iterable):
			from collections import deque
			from itertools import izip
			a = iter(iterable)
			b = deque(iterable)
			b.popleft()
			return izip(a,b)

		width, height = img.shape[0], img.shape[1]
		l1 = np.linspace(0, height, self.shape[1]+1)
		l2 = np.linspace(0, width, self.shape[0]+1)
		return [ ( Point(int(x1),int(y1)), Point(int(x2),int(y2)) ) for x1,x2 in pairwise(l1) for y1,y2 in pairwise(l2) ]

	def getROIs(self, img):
		rects = self.getRects(img)
		for p,q in rects:
			yield img[p.y:q.y, p.x:q.x]

	def compute(self, img):
		return self.getRects(img)
		

def extract_roi(filename, output_filename='output.jpg'):
	original_img = cv2.imread(filename)

	q = QEF(1,(3,1))
	rects = q.compute(original_img)
	print original_img.shape
	#for roi in q.getROIs(original_img):
		#cv2.imshow("ha", roi)
		#cv2.waitKey()

	for p,q in q.getRects(original_img):
		print p,q
		cv2.rectangle(original_img, p, q, color=(0,0,255), thickness=5)

	#cv2.imshow("ha", original_img)
	cv2.imwrite("gg.jpg", original_img)
	cv2.waitKey()

if __name__ == '__main__':
	import sys
	filename = sys.argv[1]
	if len(sys.argv) == 3:
		extract_roi(filename, sys.argv[2])
	else:
		extract_roi(filename)

