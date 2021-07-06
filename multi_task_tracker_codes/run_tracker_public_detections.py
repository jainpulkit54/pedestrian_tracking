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
from networks_reid import *
from utils import *
from PIL import Image, ImageDraw
#matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from v2_mytracker_public_detections import *

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type = str, default = "/home/pulkit/Datasets/MOT_Datasets/Detection_Datasets/MOT17Det/train/MOT17-13/img1/", help = "Path to the dataset folder")
parser.add_argument("--public_detections_file", type = str, default = "/home/pulkit/Datasets/MOT_Datasets/MOT17Labels/train/MOT17-13-FRCNN/det/det.txt", help = "Path to the public detections file for the corresponding video sequence")
parser.add_argument("--weights_path_reid", type = str, default = "checkpoints_mars+market1501_batchhard_softplus/model_epoch_4.pth", help = "Path to the reid weight file")
parser.add_argument("--lambda_iou", type = float, default = 0.4, help = "The weightage to give to IOU score for tracking")
parser.add_argument("--lambda_reid", type = float, default = 0.6, help = "The weightage to give to REID score for tracking")
parser.add_argument("--iou_thresh", type = float, default = 0.3, help = "IOU threshold for tracking")
parser.add_argument("--reid_cosine_sim_thresh", type = float, default = 0.6, help = "Threshold for cosine similarity between track and detection before which they are considered to be different")
parser.add_argument("--display", type = str, default = 'False', help = "Set this to True if want to visualize detections on image")
parser.add_argument("--tracks_folder_name", type = str, default = "outputs/MOT17/tracks_public_detections2", help = "Enter the name of the folder where you want to save the tracks obtained")
parser.add_argument("--tracks_file_name", type = str, default = "MOT17-13-FRCNN.txt", help = "Enter the file name in which you want to store the object trajectories")
parser.add_argument("--output_images_folder_name", type = str, default = "outputs/MOT17/public_detections2/MOT17-13_public_detections2", help = "Enter the folder name in which you want to save the images with tracks")
args = parser.parse_args()

if args.display == 'False':
	args.display = False
else:
	args.display = True

os.makedirs(args.tracks_folder_name, exist_ok = True)
file_name = args.tracks_folder_name + '/' + args.tracks_file_name
file_out = open(file_name, 'w+')
os.makedirs(args.output_images_folder_name, exist_ok = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Initialization
person_reid = EmbeddingNet()
# Loading the person reid checkpoint weights
checkpoint_reid = torch.load(args.weights_path_reid)
person_reid_parameters = checkpoint_reid['state_dict']
person_reid.load_state_dict(person_reid_parameters)
person_reid.to(device)
person_reid.eval()

images_names = sorted(os.listdir(args.image_folder))

# Instantiating the Tracker
tracker = Tracker(person_reid, args.lambda_iou, args.lambda_reid, args.iou_thresh, args.reid_cosine_sim_thresh, args.public_detections_file) # The instance of the person reid network will be passed

#plt.ion() # Interative Mode On
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)
colors = np.random.rand(512,3) # used only for displaying the bounding box trackers

for frame_num, image_name in enumerate(images_names):

	frame_num += 1
	img_path = args.image_folder + image_name
	img = Image.open(img_path).convert('RGB')
	ax.imshow(img)
	ax.set_axis_off()

	tic = time.time()
	
	with torch.no_grad():
		trackers = tracker.step(img, frame_num)

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
		plt.text(x = x_min, y = y_min, s = str('%d'%object_num), fontsize = 8, bbox = dict(facecolor = 'yellow', alpha = 0.5))
		print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1'%((frame_num), object_num, x_min, y_min, width, height),file = file_out)
		
	fig.savefig(args.output_images_folder_name + '/' + str(frame_num).zfill(6) + '.jpg', bbox_inches = 'tight', pad_inches = 0)

	# if index == 0:
	# 	break

	#plt.draw()
	ax.cla()
	#fig.canvas.flush_events()