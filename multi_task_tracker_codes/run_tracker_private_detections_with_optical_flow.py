import os
import cv2
import time
import torch
import argparse
import matplotlib
import numpy as np
import matplotlib.patches as patches
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import *
from networks import *
from networks_reid import *
from utils import *
from mytracker_private_detections_with_optical_flow import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 1, help = "Image Batch Size")
parser.add_argument("--image_folder", type = str, default = "/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-02/img1/", help = "Path to the dataset folder")
parser.add_argument("--network_config", type = str, default = "config/yolov3-coco2014.cfg", help = "Patch to the file containing the network definition")
parser.add_argument("--use_coco2014_weights", type = int, default = 1, help = "Set this to 1 if you want to use the pretrained COCO2014 weights else use the custom weights file")
parser.add_argument("--coco2014_weights_path_detector", type = str, default = "checkpoints_coco2014/yolov3.weights", help = "Path to the weights file")
parser.add_argument("--custom_weights_path_detector", type = str, default = "checkpoints_800_mot1720Det/model_epoch_29.pth", help = "Path to the weights file")
parser.add_argument("--weights_path_reid", type = str, default = "checkpoints_mars_batchhard/model_epoch_25.pth", help = "Path to the reid weight file")
parser.add_argument("--lambda_iou", type = float, default = 0.4, help = "The weightage to give to IOU score for tracking")
parser.add_argument("--lambda_reid", type = float, default = 0.6, help = "The weightage to give to REID score for tracking")
parser.add_argument("--class_path", type = str, default = "data/classes.names", help = "Path to the class label file")
parser.add_argument("--conf_thresh", type = float, default = 0.5, help = "Object Confidence Threshold")
parser.add_argument("--nms_thresh", type = float, default = 0.5, help = "IOU threshold for Non-Maximum Suppression")
parser.add_argument("--iou_thresh", type = float, default = 0.3, help = "IOU threshold for tracking")
parser.add_argument("--reid_cosine_sim_thresh", type = float, default = 0.6, help = "Threshold for cosine similarity between track and detection before which they are considered to be different")
parser.add_argument("--n_cpu", type = int, default = 0, help = "Number of CPU threads to use for batch generation")
parser.add_argument("--inp_img_size", type = int, default = 800, help = "Dimension of input image to the network")
parser.add_argument("--tracks_folder_name", type = str, default = "outputs/MOT17/tracks_private_detections", help = "Enter the name of the folder where you want to save the tracks obtained")
parser.add_argument("--tracks_file_name", type = str, default = "MOT17-02.txt", help = "Enter the file name in which you want to store the object trajectories")
parser.add_argument("--output_images_folder_name", type = str, default = "outputs/MOT17/private_detections/MOT17-02", help = "Enter the folder name in which you want to save the images with tracks")
args = parser.parse_args()

os.makedirs(args.tracks_folder_name, exist_ok = True)
file_name = args.tracks_folder_name + '/' + args.tracks_file_name
file_out = open(file_name, 'w+')
os.makedirs(args.output_images_folder_name, exist_ok = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Initialization
model_config = args.network_config
img_size = args.inp_img_size
object_detector = Darknet(model_config, img_size)
person_reid = EmbeddingNet()

# Loading the object detector checkpoint weights
if args.use_coco2014_weights:
	object_detector.load_darknet_weights(args.coco2014_weights_path_detector)
else:
	checkpoint_detector = torch.load(args.custom_weights_path_detector)
	object_detector_parameters = checkpoint_detector['model_state_dict']
	object_detector.load_state_dict(object_detector_parameters)

object_detector.to(device)
object_detector.eval()

# Loading the person reid checkpoint weights
checkpoint_reid = torch.load(args.weights_path_reid)
person_reid_parameters = checkpoint_reid['state_dict']
person_reid.load_state_dict(person_reid_parameters)
person_reid.to(device)
person_reid.eval()

images = ImageFolder(args.image_folder, img_size)
dataloader = DataLoader(images, batch_size = 1, shuffle = False, num_workers = args.n_cpu)

with open(args.class_path, 'r') as class_name_file:
	names = class_name_file.readlines()

class_names = []
for name in names:
	class_names.append(name.rstrip().lstrip())

images_names = sorted(os.listdir(args.image_folder))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Instantiating the Tracker
tracker = Tracker(args.lambda_iou, args.lambda_reid, object_detector, args.conf_thresh, args.nms_thresh, person_reid, args.iou_thresh, args.reid_cosine_sim_thresh) # The instances of the YOLOv3 object detector and person reid network will be passed
colors = np.random.randint(0, 255, (512, 3)) # used only for displaying the bounding box trackers

for index, image in enumerate(dataloader):

	if index == 0:
		# Finding the bounding boxes in the first frame, which will be used to find the keypoints or the feature points
		# on which the Lucan Kanade Optical Flow algorithm will work
		img = image.type(Tensor)
		old_orig_image = cv2.imread(args.image_folder + images_names[index])
		
		tic = time.time()
		with torch.no_grad():
			trackers = tracker.step(img, old_orig_image, False)
		toc = time.time()
		print('Processing Time:', (toc - tic))

		for track in trackers:
		
			x_min = int(track[0])
			y_min = int(track[1])
			x_max = int(track[2])
			y_max = int(track[3])
			width = x_max - x_min
			height = y_max - y_min
			object_num = int(track[4])
			print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1'%((index+1), object_num, x_min, y_min, width, height),file = file_out)

		# Create a mask image for drawing purposes
		# mask = np.zeros_like(old_orig_image)
		# old_trackers = trackers

	else:

		img = image.type(Tensor)
		current_orig_image = cv2.imread(args.image_folder + images_names[index])
		
		tic = time.time()
		with torch.no_grad():
			trackers = tracker.step(img, current_orig_image, True)
		toc = time.time()
		print('Processing Time:', (toc - tic))

		for track in trackers:
			x_min = int(track[0])
			y_min = int(track[1])
			x_max = int(track[2])
			y_max = int(track[3])
			width = x_max - x_min
			height = y_max - y_min
			object_num = int(track[4])

			pt1 = (x_min, y_min)
			pt2 = (x_max, y_max)
			current_orig_image = cv2.rectangle(current_orig_image, pt1, pt2, colors[object_num % 512, :].tolist(), 2)
			(text_width, text_height) = cv2.getTextSize(str('%d'%object_num), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.50, thickness = 1)[0]
			box_coords = ((x_min, y_min), (x_min + text_width + 2, y_min - text_height - 2))
			cv2.rectangle(current_orig_image, box_coords[0], box_coords[1], (0, 255, 255), cv2.FILLED)
			cv2.putText(img = current_orig_image, text = str('%d'%object_num), org = (x_min, y_min), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
						fontScale = 0.50, color = colors[object_num % 512, :].tolist(), thickness = 1)
			print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1'%((index+1), object_num, x_min, y_min, width, height),file = file_out)

		# if len(old_trackers) >= len(trackers):
		# 	for track_old in old_trackers:
		# 		x_center_old = int((track_old[0] + track_old[2])/2)
		# 		y_center_old = int((track_old[1] + track_old[3])/2)
		# 		id_old = int(track_old[4])
		# 		for track_new in trackers:
		# 			x_center_new = int((track_new[0] + track_new[2])/2)
		# 			y_center_new = int((track_new[1] + track_new[3])/2)
		# 			id_new = int(track_new[4])
		# 			if id_old == id_new:
		# 				mask = cv2.line(mask, (x_center_old, y_center_old), (x_center_new, y_center_new), (255, 0, 0), 2)
		# 				frame = cv2.circle(current_orig_image, (x_center_new, y_center_new), 5, (255, 0, 0), thickness = -1)

		# elif len(trackers) >= len(old_trackers):
		# 	for track_new in trackers:
		# 		x_center_new = int((track_new[0] + track_new[2])/2)
		# 		y_center_new = int((track_new[1] + track_new[3])/2)
		# 		id_new = int(track_new[4])
		# 		for track_old in old_trackers:
		# 			x_center_old = int((track_old[0] + track_old[2])/2)
		# 			y_center_old = int((track_old[1] + track_old[3])/2)
		# 			id_old = int(track_old[4])
		# 			if id_old == id_new:
		# 				mask = cv2.line(mask, (x_center_old, y_center_old), (x_center_new, y_center_new), (255, 0, 0), 2)
		# 				frame = cv2.circle(current_orig_image, (x_center_new, y_center_new), 5, (255, 0, 0), thickness = -1)

		# im = cv2.add(frame, mask)
		# cv2.imwrite(args.output_images_folder_name + '/' + str(index + 1).zfill(6) + '.jpg', im)
		# old_trackers = trackers		
		cv2.imwrite(args.output_images_folder_name + '/' + str(index + 1).zfill(6) + '.jpg', current_orig_image)

	# if index == 2:
	# 	break