import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_path', type = str, default = 'NCETIS.avi', help = 'Enter the video path whose frames you want to extract')
parser.add_argument('--output_image_path', type = str, default = 'data/ncetis/images/', help = 'The folder path where images with detections are stored')
args = parser.parse_args()

video_path = args.input_video_path
images_path = args.output_image_path
os.makedirs(images_path, exist_ok = True)

cap = cv2.VideoCapture(video_path)
index = 0

while(cap.isOpened()):
	ret, frame = cap.read()
	frame = cv2.resize(frame, (1920, 1080))
	index = index + 1
	filename = images_path + str(index).zfill(6) + '.jpg'
	cv2.imwrite(filename, frame)

cap.release()
cv2.destroyAllWindows()