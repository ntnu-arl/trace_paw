# This code is copied from the github repository: https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

'''
Sample Command:-
python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100
'''
import numpy as np
from SmartPawDataset.sensors.Oakd_Lite.utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys
import os





ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagefolder", required=True, help="path to folder of images containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())



print("Loading images...")
path = args["imagefolder"]
goodImg = 0
imgNr = 0
for test_image in os.listdir(path):
	file = os.path.join(path,test_image)
	image = cv2.imread(file)
	h,w,_ = image.shape
	width=600
	height = int(width*(h/w))
	image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


	# verify that the supplied ArUCo tag exists and is supported by OpenCV
	if ARUCO_DICT.get(args["type"], None) is None:
		print(f"ArUCo tag type '{args['type']}' is not supported")
		sys.exit(0)

	# load the ArUCo dictionary, grab the ArUCo parameters, and detect
	# the markers
	# print("Detecting '{}' tags....".format(args["type"]))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
	arucoParams = cv2.aruco.DetectorParameters_create()
	corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
	if len(ids) == 2:
		goodImg += 1
	if imgNr % 50 == 0:
		print(imgNr)
		print("Number of good images: ", goodImg)
	imgNr += 1
	# detected_markers = aruco_display(corners, ids, rejected, image)
	# cv2.imshow("Image", detected_markers)

	# # # Uncomment to save
	# # cv2.imwrite("output_sample.png",detected_markers)

	# cv2.waitKey(0)
print("Number of good images: ", goodImg)