from threading import Thread
import depthai as dai
import time
import cv2
import numpy as np
import math

from sensors.Oakd_Lite.aruco_pose import aruco_pose 
from sensors.Oakd_Lite.utils import ARUCO_DICT

class OAKd_Camera(Thread):
    def __init__(self):
        """This class will spawn a thread controlling the connection with the camera,
        provide output frames, and augmention with pose estimation based on Aruco markers.

        """
        Thread.__init__(self)
        self.daemon = True
        self.start()
        self.running = True
        self.displayPose=False
        self.calMatrix = None
        self.distCoeff = None
        self.aruco_dict_type = None
        self.frame = None
        self.ids = None

    def run(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define source and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        xoutVideo = self.pipeline.create(dai.node.XLinkOut)

        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(True)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Linking
        camRgb.video.link(xoutVideo.input)

        # # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
            video = device.getOutputQueue('video')
            while self.running:
                videoframe = video.get()
                self.frame = videoframe.getCvFrame()


    def enablePoseEstimation(self, pathCalMatrix, pathDistCoeff, aruco_dict_type):
        self.displayPose=True
        self.calMatrix = np.load(pathCalMatrix)
        self.distCoeff = np.load(pathDistCoeff)
        self.aruco_dict_type = ARUCO_DICT[aruco_dict_type]


    def disablePoseEstimation(self):
        self.displayPose=False
        self.calMatrix = None
        self.distCoeff = None
        self.aruco_dict_type = None

    def getFrame(self):
        return self.frame


    def getPoseFrame(self):
        aruco = aruco_pose()
        self.ids = aruco.getNumberTags(self.frame)
        # poseDict = aruco.estimateArucoPose(self.frame)
        # relativePose = aruco.estimateRelativePose(poseDict)
        # if relativePose is not None:
        #     for i in range(len(relativePose)):
        #         relativePoseDeg = np.array((3,1))
        #         relativePoseDeg[i] = math.degrees(relativePose[i])
        #     print("Rotation vector: ", relativePoseDeg)
        poseFrame = aruco.showArucoPose(self.frame)
        return poseFrame

    def getNumberTags(self):
        
        if self.ids is not None:
            return self.ids
        return 0

    def stop(self):
        self.running = False


if __name__ == '__main__':
    camera = OAKd_Camera()
    counter = 0
    path = "calibration/cal_images/"
    time.sleep(5)
    while True:
        frame = camera.getFrame()
        cv2.imshow("camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            camera.enablePoseEstimation(
                pathCalMatrix = 'calibration/calibration_matrix.npy',
                pathDistCoeff = 'calibration/distortion_coefficients.npy',
                aruco_dict_type = 'DICT_5X5_100'
            )
        if key == ord('d'):
            camera.disablePoseEstimation()

        if key == ord('s'):
            # Save frame
            counter += 1
            cv2.imwrite(f'{path}image_{counter}.jpeg',frame)

        if key == ord('q'):
            break