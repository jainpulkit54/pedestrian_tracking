import os
import cv2
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

det_dataset1 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-01/img1/' # 25 fps
det_dataset1_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-01/gt/gt.txt'
images_names1 = sorted(glob.glob(det_dataset1 + '*.jpg'))
height1 = 1080
width1 = 1920

det_dataset2 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-02/img1/' # 25 fps
det_dataset2_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-02/gt/gt.txt'
images_names2 = sorted(glob.glob(det_dataset2 + '*.jpg'))
height2 = 1080
width2 = 1920

det_dataset3 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-03/img1/' # 14 fps
det_dataset3_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-03/gt/gt.txt'
images_names3 = sorted(glob.glob(det_dataset3 + '*.jpg'))
height3 = 880
width3 = 1173

det_dataset4 = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-05/img1/' # 30 fps
det_dataset4_gt = '/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT20Det/train/MOT20-05/gt/gt.txt'
images_names4 = sorted(glob.glob(det_dataset4 + '*.jpg'))
height4 = 1080
width4 = 1654

def parse_gt_file(file_path):

	with open(file_path, 'r') as file:
		rows = file.readlines()

	sorted_rows = []
	
	for row in rows:
		row = row.rstrip().lstrip()
		row_list = row.split(',')
		r = []
		for element in row_list:
			r.append(float(element.rstrip().lstrip()))
		sorted_rows.append(r)

	sorted_rows = sorted(sorted_rows)
	sorted_rows = np.array(sorted_rows)
	return sorted_rows

def create_ann_file(r, width, height, class_0, class_1):

	frame_nos = np.unique(r[:,0])

	for frame_number in frame_nos:
		
		indices = np.where(r[:,0] == frame_number)[0]
		subarray = r[np.ix_(indices, [2,3,4,5,7])]
		
		for i in range(subarray.shape[0]):
			row = subarray[i,:]
			box_xmin = row[0]; box_ymin = row[1]; box_width = row[2]; box_height = row[3]; cls = row[4]
			box_xmin = np.clip(box_xmin, 0, width - 1)
			box_ymin = np.clip(box_ymin, 0, height - 1)
			box_xmax = box_xmin + box_width
			box_ymax = box_ymin + box_height

			if box_xmax >= width or box_ymax >= height:
				box_xmax = np.clip(box_xmax, 0, width - 1)
				box_ymax = np.clip(box_ymax, 0, height - 1)
				box_width = box_xmax - box_xmin
				box_height = box_ymax - box_ymin

			box_xmin = box_xmin/width
			box_ymin = box_ymin/height
			box_width = box_width/width
			box_height = box_height/height
			box_xmin = box_xmin + box_width/2
			box_ymin = box_ymin + box_height/2

			if cls == 1 or cls == 2 or cls == 7:
				class_0 += 1
			elif cls == 3 or cls == 4 or cls == 5 or cls == 6 or cls == 8 or cls == 12:
				class_1 += 1
			# ignoring classes 9, 10, 11, 13
			#elif cls == 8 or cls == 9 or cls == 10 or cls == 11 or cls == 12 or cls == 13:
			#	class_1 += 1
			# ignoring classes 3,4,5,6

	return class_0, class_1

r1 = parse_gt_file(det_dataset1_gt)
class_0, class_1 = create_ann_file(r1, width1, height1, class_0 = 0, class_1 = 0)

r2 = parse_gt_file(det_dataset2_gt)
class_0, class_1 = create_ann_file(r2, width2, height2, class_0 = class_0, class_1 = class_1)

r3 = parse_gt_file(det_dataset3_gt)
class_0, class_1 = create_ann_file(r3, width3, height3, class_0 = class_0, class_1 = class_1)

r4 = parse_gt_file(det_dataset4_gt)
class_0, class_1 = create_ann_file(r4, width4, height4, class_0 = class_0, class_1 = class_1)

print("Number of objects in CLASS-0 in MOT20 are:", class_0)
print("Number of objects in CLASS-1 in MOT20 are:", class_1)