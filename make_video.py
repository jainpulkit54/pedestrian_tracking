import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_path', type = str, default = 'outputs/marathon/private_detections/', help = 'The folder path where images with detections are stored')
parser.add_argument('--output_video_path', type = str, default = 'videos', help = 'The folder path where videos will be saved')
parser.add_argument('--video_name', type = str, default = 'marathon.avi', help = 'The video file name')
parser.add_argument('--video_width', type = int, default = 1920, help = 'The video output frame width')
parser.add_argument('--video_height', type = int, default = 1080, help = 'The video output frame height')
parser.add_argument('--fps', type = str, default = '30', help = 'Enter the Frame Per Second')
args = parser.parse_args()

os.makedirs(args.output_video_path, exist_ok = True)
image_path = args.input_image_path
images_names = sorted(os.listdir(image_path))
video_name = args.video_name
width = args.video_width
height = args.video_height
fps = args.fps
fps = float(fps)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter((args.output_video_path + '/' + video_name), fourcc, fps, (width, height))

img_array = []

for img_name in images_names:
	
	frame = cv2.imread(image_path + img_name)
	frame = cv2.resize(frame, (width, height))
	video.write(frame)

video.release()
cv2.destroyAllWindows()