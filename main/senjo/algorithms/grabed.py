import cv2
import math
import numpy as np

def line(x0,y0,x1,y1):
	print 'line from (%d,%d) to (%d,%d)' % (x0,y0,x1,y1)
	dx = x1 - x0
	dy = y1 - y0
	eps = 0

	y = y0
	for x in range(x0, x1):
		yield x,y
		eps += dy
		if (eps << 1) >= dx:
			y += 1
			eps -= dx

def line2(x0,y0,x1,y1):
	print 'line2 from (%d,%d) to (%d,%d)' % (x0,y0,x1,y1)
	dx = x1 - x0
	dy = y1 - y0
	eps = 0

	x = x0
	for y in range(y0, y1):
		yield x,y
		eps += dx
		if (eps << 1) >= dy:
			x += 1
			eps -= dy

def strel(l,d):
	print l,d
	def _get_rect_shape(l):
		if l % 2 == 0:
			l+=1
		return l,l
	w,h = _get_rect_shape(l)

	d = float(d)
	m = math.tan(math.radians(d))
	if d <= 45.0:
		x1 = w
		y1 = m*x1
		_line = line
	elif d <= 90.0:
		y1 = h
		x1 = y1 / m
		_line = line2
	elif d <= 180.0:
		output = strel(l, d-90.0)
		return np.rot90(output)

	m = np.zeros((w,h))
	x1,y1 = int(math.ceil(x1)), int(math.ceil(y1))
	for x,y in _line(0,0,x1,y1):
		m[x][y] = 1
	m = np.rot90(m)
	output = np.zeros(m.shape, dtype=np.uint8)
	output[0:w/2+1, h/2:h] = m[w/2:w+1, 0:h/2+1]
	output[w/2:w+1, 0:h/2+1] = np.rot90(output[0:w/2+1, h/2:h],2)

	return output
'''
print strel(10,0)
print strel(10,15)
print strel(10,30)
print strel(10,45)
print strel(10,60)
print strel(10,75)
print strel(10,90)
print strel(10,120)
print strel(10,150)
print strel(10,180)
'''
def gen_ees(max_axis, l0, r):
	sizes = []
	l = l0
	while l <= max_axis:
		sizes.append(l)
		l = l*2
	print 'sizes', sizes
	angles = np.linspace(0,180,r)
	print 'angles', angles

	return [ (l,a,strel(l,a)) for a in angles for l in sizes ]

for ee in gen_ees(8,2,4):
	print ee

def default_border_extractor(img):
	#img = cv2.blur(img,(5,5))
	img = cv2.Canny(img,50,100)
	#img = (255-img)
	return img

def grabed(img, scales=1, f=0.5, max_axis=128, l0=3, r=12, border_extractor=default_border_extractor, output_file=None):
	def _get_ee_families():
		from collections import defaultdict
		from operator import itemgetter

		EEs = gen_ees(max_axis, l0, r)
		# Group by angle
		d = defaultdict(list)
		for ee in EEs:
			d[ee[1]].append((ee[0],ee[2]))
		EE_families = d.items()
		EE_families.sort(key=itemgetter(0))
		for a, ee_group in EE_families:
			ee_group.sort(key=itemgetter(0))
		return EE_families
	print 'grabed', scales, f, max_axis, l0, r, border_extractor
	ps = []
	for s in xrange(1,scales+1):
		ir = cv2.resize(img, (0,0), fx=f, fy=f)
		borders = border_extractor(ir)
		initial_area = np.count_nonzero(borders)
		prev_area = initial_area

		_vstack = []
		initial_area = float(initial_area)
		for a,ee_family in _get_ee_families():
			_stack = []
			_stack.append(borders)

			acc = 0.0
			for l, ee in ee_family:
				io = cv2.morphologyEx(borders, cv2.MORPH_OPEN, ee)
				tmp_area = np.count_nonzero(io)
				ps.append( (prev_area-tmp_area)/initial_area )
				acc = acc + (prev_area-tmp_area)/initial_area

				prev_area = tmp_area
				_stack.append(io)
			print 'acc',acc

			_vstack.append(np.hstack(_stack))
		if output_file is not None:
			cv2.imwrite(output_file, (255-np.vstack(_vstack)))
			print 'written', output_file
	return ps

img_path = 'image_0003.jpg'
#img_path = 'not_so_similar.jpg'
img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img = default_border_extractor(img)

print grabed(img, output_file='bla2.jpg')
