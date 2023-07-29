import cv2
import numpy as np
import math


"""
    This script is used to measure the angle between two lines, drawn by placing 4 points on the image.
    The reason is to measure the relative angle between wagon and its aruco tag.

    angle in Radians:  0.03350700778086374
    angle in Degrees:  2

    angle in Radians:  0.03981806849665749
    angle in Degrees:  2

    angle in Radians:  0.033263265648636166
    angle in Degrees:  2

    angle in Radians:  0.03521767579480664
    angle in Degrees:  2

    angle in Radians:  0.03431953614088705
    angle in Degrees:  2

    angle in Radians:  0.02681643840973599
    angle in Degrees:  2

    angle in Radians:  0.03600944343511573
    angle in Degrees:  2

    Mean : 0.03413 rads

"""



# Set up the file paths to our data set 
filepathAruco = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/outputs/cleanData/Oakd_lite/'
file = '1665404316776' + '.jpeg'

image = cv2.imread(filepathAruco + file)
pointsList = []

def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y), 2, (0,0,255), cv2.FILLED)
        pointsList.append([x,y])
        print(pointsList)

def gradient(pt1, pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

def getAngle(image):
    if len(pointsList) == 4:
        cv2.line(image, pointsList[0], pointsList[1], (0, 255, 0), 2)
        cv2.line(image, pointsList[2], pointsList[3], (0, 255, 0), 2)
        m1 = gradient(pointsList[0], pointsList[1])
        m2 = gradient(pointsList[2], pointsList[3])
        angR = math.atan((m1-m2)/(1+m1*m2))
        angD = round(math.degrees(angR))
        print("angle in Radians: ", angR)
        print("angle in Degrees: ", angD)
    return image


while True:
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', mousePoints)
    image = getAngle(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        image = cv2.imread(filepathAruco + file)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break