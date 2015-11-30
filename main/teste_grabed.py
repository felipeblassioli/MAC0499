from senjo.algorithms.grabed import grabed
import cv2

img_path = 'bota.jpg'
img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
grabed(img, output_file='bota_grabed.png')
