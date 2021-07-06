import os
import cv2
import torch
import numpy as np
from utils import *
from PIL import Image
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

class Tracker:

	# We are here considering only "Pedestrian Tracking"
	
	def __init__(self, lambda_iou, lambda_reid, object_detector, conf_thresh, nms_thresh, person_reid, iou_thresh, reid_cosine_sim_thresh):
		
		self.object_detector = object_detector # This will be the instance to the YOLOv3 Object Detector
		self.person_reid = person_reid # This will be the instance to the Person Reidentification network
		self.detection_object_threshold = conf_thresh # YOLOv3 score threshold for detections
		self.detection_nms_threshold = nms_thresh # NMS threshold for detection
		self.iou_threshold = iou_thresh # IOU threshold for tracking
		self.reid_cosine_sim_threshold = reid_cosine_sim_thresh # Threshold for cosine distance between track and detection after which they are considered to be different
		self.lambda_iou = lambda_iou
		self.lambda_reid = lambda_reid		
		#---------------------------------------------------------
		# Settings for Motion Model and Camera Motion Compensation
		#---------------------------------------------------------

		# Arrays that will be used for tracking purposes
		self.trackers = []
		self.track_id = []
		self.track_id_count = 1
		self.max_age = 200 # The number of frames for which each tracker will be kept alive before deletion in case of missing detections

	def iou_individual(self, ground_truth, prediction):

		# This function calculates the intersection over union between two boxes and returns the same

		x1 = max(ground_truth[0], prediction[0])
		y1 = max(ground_truth[1], prediction[1])
		x2 = min(ground_truth[2], prediction[2])
		y2 = min(ground_truth[3], prediction[3])

		intersection_area = max(0,(x2 - x1)) * max(0,(y2 - y1))
		area_gt = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
		area_pred = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
		union_area = area_gt + area_pred - intersection_area + 1e-16
		iou = intersection_area/union_area

		return iou

	def associate_detections(self, detections, trackers):

		no_of_detections = len(detections)
		no_of_trackers = len(trackers)

		if no_of_trackers == 0:
			return np.empty([0,2], dtype = np.int32), np.arange(no_of_detections), np.arange(no_of_trackers)

		if no_of_detections == 0:
			return np.empty([0,2], dtype = np.int32), np.arange(no_of_detections), np.arange(no_of_trackers)

		cosine_similarity_matrix = np.zeros([no_of_detections, no_of_trackers], dtype = np.float32)
		iou_matrix = np.zeros([no_of_detections, no_of_trackers], dtype = np.float32)

		for index_det, detection in enumerate(detections):
			for index_track, tracker in enumerate(trackers):
				iou_matrix[index_det, index_track] = self.iou_individual(detection[0:4], tracker['coordinates'])
				cosine_similarity_matrix[index_det, index_track] = 1 - distance.cosine(detection[5:133], tracker['features'])

		combined_matrix = self.lambda_iou * iou_matrix + self.lambda_reid * cosine_similarity_matrix
		row_ind, col_ind = linear_sum_assignment(-1*combined_matrix)
		matched_indices = np.array(list(zip(row_ind, col_ind)))

		unmatched_detections = []
		# Now finding out the unmatched detections for which the new trackers need to be created
		for index_det, detection in enumerate(detections):
			if index_det not in matched_indices[:,0]:
				unmatched_detections.append(index_det)

		unmatched_trackers = []
		# Now finding out the unmatched trackers
		for index_track, tracker in enumerate(trackers):
			if index_track not in matched_indices[:,1]:
				unmatched_trackers.append(index_track)

		# Now filtering out the matches that have IOU less than the IOU threshold and that have cosine similarity less than the cosine similarity threshold
		matches = []
		for m in matched_indices:
			if (iou_matrix[m[0], m[1]] < self.iou_threshold) or (cosine_similarity_matrix[m[0], m[1]] < self.reid_cosine_sim_threshold):
				unmatched_detections.append(m[0])
				unmatched_trackers.append(m[1])
			else:
				matches.append(m.reshape(1,2))

		if len(matches) == 0:
			matches = np.empty([0,2], dtype = np.int32)
		else:
			matches = np.concatenate(matches, axis = 0) # This command basically flattens the 2D array into 1D array

		return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

	def extract_embeddings(self, imgs):

		with torch.no_grad():
			embeddings = self.person_reid.get_embeddings(imgs)

		embeddings = embeddings.cpu().numpy()
		return embeddings

	# This function will be called every time step with the current image frame
	def step(self, frame, orig_image):
		
		#########################
		# New Detections

		detections = self.object_detector(frame)
		detections = non_max_suppression(detections, self.detection_object_threshold, self.detection_nms_threshold)
		orig_image_size = orig_image.size
		img_size = frame.shape[2] # Since the frame is a square image
		
		try:
			detections = detections[0]
			indices = (detections[:,-1] == 0) # Filtering out the detections corresponding only to the "person" class
			detections = detections[indices, :]
			bbox = detections[:,0:4]/img_size # [x_min, y_min, x_max, y_max] in Normalized (0-1) form
			score = detections[:,5].view(-1,1) # This gives the probability of "person" class
			bbox[:,0] = bbox[:,0] * orig_image_size[0]
			bbox[:,1] = bbox[:,1] * orig_image_size[1]
			bbox[:,2] = bbox[:,2] * orig_image_size[0]
			bbox[:,3] = bbox[:,3] * orig_image_size[1]
			dets = np.concatenate((np.array(bbox), np.array(score)), axis = 1)

			# Code for extracting the cropped detection portions from the image, resizing it to 256*128 for Person Reid
			# and storing the extracted image portions in a dictionary
			imgs = []
			for dimensions in dets:
				left = dimensions[0]
				top = dimensions[1]
				right = dimensions[2]
				bottom = dimensions[3]
				im = orig_image.crop((left, top, right, bottom))
				im = im.resize((128, 256))
				im = np.array(im)
				im = im/255.0
				im = np.transpose(im, (2,0,1))
				imgs.append(im)

			imgs = np.array(imgs)
			imgs = torch.from_numpy(imgs).float()
			
			if torch.cuda.is_available():
				imgs = imgs.cuda()
			
			embeddings = self.extract_embeddings(imgs)
			dets = np.concatenate((dets, embeddings), axis = 1)
		
		except:

			# 133 dimension vector (4 for object coordinates, 1 for detection probabilty, 128 for detection embeddings)
			dets = np.empty([0,133], dtype = np.float32)
			
		#########################

		#########################
		# Predict Tracks

		# The prediction of the track will be done when some correction is applied based on the previous track i.e.,
		# like making using of "image alignment" for Camera Motion Compensation (CMC) or using "Motion Model" like
		# "Constant Velocity Assumption" (CVA) for videos where there is significant motion between the two consecutive
		# frames

		#########################

		#########################
		# Data Association

		matches, unmatched_detections, unmatched_trackers = self.associate_detections(dets, self.trackers)
		
		# Handling the case of Dead Trackers
		# For all the trackers that have matched with the previous detections, set "time_since_update" to "0"
		
		for match in matches:
			index_det = match[0]
			index_track = match[1]
			self.trackers[index_track]['coordinates'] = dets[index_det, 0:4]
			self.trackers[index_track]['features'] = dets[index_det, 5:133]
			self.trackers[index_track]['track_id'] = self.track_id[index_track]
			self.trackers[index_track]['time_since_update'] = 0

		for index_track in unmatched_trackers:
			self.trackers[index_track]['time_since_update'] += 1

		#########################

		#########################
		# Create New Tracks

		# Creating and initializing new trackers for unmatched detections
		for index_det in unmatched_detections:
			self.trackers.append({'coordinates': dets[index_det, 0:4], 'features': dets[index_det, 5:133], 'track_id': self.track_id_count, 'time_since_update': 0})
			self.track_id.append(self.track_id_count)
			self.track_id_count += 1
			
		#########################

		#########################
		# Generate Results
		
		ret = []
		total_trackers = len(self.trackers)
		for track in reversed(self.trackers):

			total_trackers -= 1
			# Removing the dead tracker according to the max age parameter
			if (track['time_since_update'] > self.max_age):
			 	self.trackers.pop(total_trackers)
			 	self.track_id.pop(total_trackers)
			
			if (track['time_since_update'] == 0):
				ret.append(np.concatenate((track['coordinates'], [track['track_id']])).reshape(1, -1))

		if len(ret) > 0:
			return np.concatenate(ret)
		else:
			return np.empty([0,5])

		#########################