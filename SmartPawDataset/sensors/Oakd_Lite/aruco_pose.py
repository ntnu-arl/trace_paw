import numpy as np
import argparse
import cv2
import sys
import math
from scipy.spatial.transform import Rotation

# from sensors.Oakd_Lite.utils import ARUCO_DICT
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

class aruco_pose:
    def __init__(self, 
                aruco_dict_type= "DICT_7X7_50",
                pathCalMatrix = "E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/calibration/calibration_matrix.npy",
                pathDistCoeff = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/calibration/distortion_coefficients.npy'):
        """
        This class will provide methods for interacting with aruco tags, such as generating them, detecting them in images and retreiving their pose in images.

        Args:
            aruco_dict_type (string)    : Containing the dictionary key for the Aruco type 
            pathCalMatrix   (string)    : Containing the path to the calibration matrix file
            pathDistCoeff   (string)    : Containing the path to the distortion coefficient matrix
        """

        if ARUCO_DICT.get(aruco_dict_type, None) is None:
            print(f"ArUCo tag type '{aruco_dict_type}' is not supported")
            sys.exit(0)
        

        self.arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dict_type])
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        

        self.arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
        self.markerSize = 0.02
        self.objPoints = np.float32([(-self.markerSize/2, self.markerSize/2, 0), 
                        (self.markerSize/2, self.markerSize/2, 0),
                        (self.markerSize/2, -self.markerSize/2, 0),
                        (-self.markerSize/2, -self.markerSize/2, 0)])

        self.calMatrix = np.load(pathCalMatrix)
        self.distCoeff = np.load(pathDistCoeff)
        self.frame_offset = math.pi/2
        self.forceSensor_offset = math.radians(-64.5)
        # self.hwoffsetWagon_zAxis = math.radians(-2)
        # self.hwOffsetPaw_xAxis = -0.0469322433
        # self.hwOrientationOffset_xAxis = -math.pi/2


    def detectArucoMarkers(self, image):
        """
        This method detects the aruco markers and returns the pixel coordinates of the corners, the unique ID of the tag, and the rejected tags.

        Args:
            image (numpy.ndarray)   : frame containing the image 
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = self.arucoDetector.detectMarkers(gray)
        # corners, ids, rejected, recovered = self.arucoDetector.refineDetectedMarkers(gray, board, corners, ids, rejected, self.calMatrix, self.distCoeff)
        return corners, ids, rejected

    def showDetectedMarkers(self, image):
        """
        This method detects the aruco markers and displays an image with them highlighted and IDed.

        Args:
            image (numpy.ndarray)   : frame containing the image 
        """
        corners, ids, rejected = self.detectArucoMarkers(image)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("image", image)
        cv2.waitKey(0)



    def estimateArucoPose(self, image, extraRot=False):
        """
        This method detects the aruco markers, estimates the pose of the markers and returns it in a dictionary

        Args:
            image (numpy.ndarray)   : frame containing the image 

        Return:
            poseDict (dictionary) : dictionary containing the pose of the markers (rvec, tvec), rvec is stored as a Rodrigues vector
        """


        corners, ids, _ = self.detectArucoMarkers(image)
        poseDict = {}
        if isinstance(ids, np.ndarray):
            if len(ids) >= 2:
                for i in range(len(ids)):
                    rvec = np.float32([1]*3)
                    tvec = np.float32([1]*3)
                    cv2.solvePnP(self.objPoints, corners[i], self.calMatrix, self.distCoeff, rvec, tvec) # Writes to rvec and tvec

                    #Correct the orientation, by rotating +90* about the x axis.
                    rmat  = np.eye(3)
                    rmat, _ = cv2.Rodrigues(rvec)

                    rmat = rmat @ np.array([
                                    [1,                    0,                               0],
                                    [0,                    math.cos(self.frame_offset),    -math.sin(self.frame_offset)],
                                    [0,                    math.sin(self.frame_offset),    math.cos(self.frame_offset)]])


                    # Check if ID correpsonds to TestBed, and correct for force sensor orientation about the z-axis
                    if ids[i] == 0:
                        rmat = rmat @ np.array([    [math.cos(self.forceSensor_offset),     -math.sin(self.forceSensor_offset),     0],
                                                    [math.sin(self.forceSensor_offset),     math.cos(self.forceSensor_offset),      0],
                                                    [0 ,                    0,                      1]])
                    if extraRot:
                        # Check if ID correpsonds to Paw, and correct for force sensor orientation about the z-axis
                        if ids[i] == 1:
                            rmat = rmat @ np.array([    [math.cos(np.pi/2),     -math.sin(np.pi/2),     0],
                                                        [math.sin(np.pi/2),     math.cos(np.pi/2),      0],
                                                        [0 ,                    0,                      1]])

                    rvec, _ = cv2.Rodrigues(src=rmat)
                    poseDict[f"{ids[i][0]}"] = [rvec, tvec]

        return poseDict



    def showArucoPose(self, image):
        """
        This method detects the aruco markers, estimates the pose of the markers and displays it.
        Red:    x-axis,
        Green:  y-axis,
        Blue:   z-axis 

        Args:
            image (numpy.ndarray)   : frame containing the image 
        """
        corners, ids,_ = self.detectArucoMarkers(image)
        # print(ids)
        poseDict = self.estimateArucoPose(image)

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids) 
        for key in poseDict.keys():
            rvec = poseDict[key][0]
            tvec = poseDict[key][1]
            cv2.drawFrameAxes(image, self.calMatrix, self.distCoeff, rvec, tvec, 0.01) 
        # cv2.imshow("Orientation_frame", image)
        return image

    def getNumberTags(self, image):
        corners, ids,_ = self.detectArucoMarkers(image)
        if ids is not None:
            return len(ids)
        return 0

    def estimateRelativePose(self, poseDict, 
                            testbedFrame="0", pawFrame="1"):
        """
        This method estimates the orientation of the pawFrame relative to the testbedFrame, from the acuro markers found in the image.

        Args:
            image (numpy.ndarray)   : frame containing the image 
            testbedFrame (string)   : String containing the key for the wagon frame in this classes poseDict
            pawFrame (string)       : String containing the key for the paw frame in this classes poseDict

        Returns:
            rot_vec (numpy.ndarray) : List of 3 elements containing the rotations about x,y and z axis in radians.
        """

        if pawFrame in poseDict:
            # Homogenous transformation matrix between 3D points of testbed and camera. 
            # NB! It may have to be inverted 

            # Transformation of the testbeds coordinates in the camera-frame
            rot_cam_testbed = np.eye(3)
            rot_cam_testbed, _ = cv2.Rodrigues(poseDict[testbedFrame][0])
            t_cam_testbed = poseDict[testbedFrame][1].reshape((3,1))
            transform_cam_bed = np.zeros((4,4)) 
            transform_cam_bed[:3,:3] = rot_cam_testbed
            transform_cam_bed[:3,3:4] = t_cam_testbed
            transform_cam_bed[3,3] = 1

            # The inverted version : Transformation of the camera in the testbeds-frame
            # transform_bed_cam = np.zeros((4,4)) 
            # transform_bed_cam[:3,:3] = rot_cam_testbed.T
            # transform_bed_cam[:3,3:4] = -(rot_cam_testbed.T)@t_cam_testbed
            # transform_bed_cam[3,3] = 1


            # Homogenous transformation matrix between 3D points of paw and camera. 
            # Transformation of the paw coordinates in the camera-frame
            rot_cam_paw = np.eye(3)
            rot_cam_paw, _ = cv2.Rodrigues(poseDict[pawFrame][0])
            t_cam_paw = poseDict[pawFrame][1].reshape((3,1))
            # transform_cam_paw = np.zeros((4,4)) 
            # transform_cam_paw[:3,:3] = rot_cam_paw
            # transform_cam_paw[:3,3:4] = t_cam_paw
            # transform_cam_paw[3,3] = 1

            # The inverted version : Transformation of the camera in the paws-frame
            transform_paw_cam = np.zeros((4,4)) 
            transform_paw_cam[:3,:3] = rot_cam_paw.T
            transform_paw_cam[:3,3:4] = -(rot_cam_paw.T)@t_cam_paw
            transform_paw_cam[3,3] = 1


            # Calculating the Homogenous transformation between paw and testbed.
            # Transformation of the testbed coordinates in the paw-frame
            transform_paw_bed = np.matmul(transform_paw_cam ,transform_cam_bed)
            rot_paw_wagon = transform_paw_bed[:3,:3]
            rot_paw_wagon = Rotation.from_matrix(rot_paw_wagon)
            # Stored as yaw (Z), pitch (Y), roll (X) (http://www.kostasalexis.com/frame-rotations-and-representations.html)
            rot_vec_euler = rot_paw_wagon.as_euler('zyx') 
            return rot_vec_euler
        return None







import os



if __name__ == '__main__':
    arucoCMD = aruco_pose()
    folderPath= "E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/tags"
    results = []
    for test_image in os.listdir(folderPath):
        file = os.path.join(folderPath, test_image)
        image = cv2.imread(file)
        # pose = arucoCMD.estimateArucoPose(image)
        # arucoCMD.showDetectedMarkers(image)
        arucoCMD.showArucoPose(image)
        # print(arucoCMD.estimateRelativePose(pose))
        cv2.waitKey(0)



