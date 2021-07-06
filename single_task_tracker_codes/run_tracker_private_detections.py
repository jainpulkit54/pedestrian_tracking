import os
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
from utils import *
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from mytracker_private_detections import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 1, help = "Image Batch Size")
parser.add_argument("--image_folder", type = str, default = "/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-13/img1/", help = "Path to the dataset folder")
parser.add_argument("--network_config", type = str, default = "config/yolov3-coco2014.cfg", help = "Patch to the file containing the network definition")
parser.add_argument("--use_coco2014_weights", type = int, default = 1, help = "Set this to 1 if you want to use the pretrained COCO2014 weights else use the custom weights file")
parser.add_argument("--coco2014_weights_path_detector", type = str, default = "checkpoints_coco2014/yolov3.weights", help = "Path to the weights file")
parser.add_argument("--custom_weights_path_detector", type = str, default = "checkpoints_800_mot1720Det/model_epoch_29.pth", help = "Path to the weights file")
parser.add_argument("--class_path", type = str, default = "data/classes.names", help = "Path to the class label file")
parser.add_argument("--conf_thresh", type = float, default = 0.5, help = "Object Confidence Threshold")
parser.add_argument("--nms_thresh", type = float, default = 0.5, help = "IOU threshold for Non-Maximum Suppression")
parser.add_argument("--iou_thresh", type = float, default = 0.3, help = "IOU threshold for tracking")
parser.add_argument("--n_cpu", type = int, default = 0, help = "Number of CPU threads to use for batch generation")
parser.add_argument("--inp_img_size", type = int, default = 800, help = "Dimension of input image to the network")
parser.add_argument("--tracks_folder_name", type = str, default = "outputs_simple/tracks_private_detections", help = "Enter the name of the folder where you want to save the tracks obtained")
parser.add_argument("--tracks_file_name", type = str, default = "MOT17-13.txt", help = "Enter the file name in which you want to store the object trajectories")
parser.add_argument("--output_images_folder_name", type = str, default = "outputs_simple/private_detections/MOT17-13", help = "Enter the folder name in which you want to save the images with tracks")

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

# Loading the object detector checkpoint weights
if args.use_coco2014_weights:
	object_detector.load_darknet_weights(args.coco2014_weights_path_detector)
else:
	checkpoint_detector = torch.load(args.custom_weights_path_detector)
	object_detector_parameters = checkpoint_detector['model_state_dict']
	object_detector.load_state_dict(object_detector_parameters)

object_detector.to(device)
object_detector.eval()

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
tracker = Tracker(object_detector, args.conf_thresh, args.nms_thresh, args.iou_thresh) # The instances of the YOLOv3 object detector will be passed

fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)

colors = np.random.rand(512,3) # used only for displaying the bounding box trackers

for index, image in enumerate(dataloader):

	img = image.type(Tensor)
	orig_image = Image.open(args.image_folder + images_names[index])
	orig_image_size = orig_image.size
	
	ax.imshow(orig_image)
	ax.set_axis_off()

	tic = time.time()
	with torch.no_grad():
		trackers = tracker.step(img, orig_image)
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
		ax.add_patch(patches.Rectangle((x_min, y_min), width, height, fill = False, lw = 3, ec = colors[(object_num % 512), :]))
		plt.text(x = x_min, y = y_min, s = str('%d'%object_num), fontsize = 8)
		print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1'%((index+1), object_num, x_min, y_min, width, height),file = file_out)
		
	fig.savefig(args.output_images_folder_name + '/' + str(index + 1).zfill(6) + '.jpg', bbox_inches = 'tight', pad_inches = 0)

	# if index == 0:
	# 	break

	ax.cla()