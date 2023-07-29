import sensors.NiclaVision.rpc_image_transfer as NiclaVision
import sensors.ForceTorqueSensor.FTSensor as FTsensor
import sensors.Oakd_Lite.camera_controls as poseCamera
import utils as utils
import os
import pandas as pd
import time
import cv2


############################################################################

# This script gathers data from multiple sorces and stores them in the output folder:
# - Image data from Nicla Vision camera sensor
# - Force and Torque data from mini45 F/T Sensor
# - Image data from Oak-d Lite camera sensor - From this we will estimate the pose of the paw relative to the ground plane


############################################################################


if __name__ == '__main__':
    record = False

    # Provide path to the desired output directory
    outputPath = "E:/Bruker/Dokumenter/Skole/Master/smart_paws/data/machined/"
    os.makedirs(outputPath, exist_ok=True)

    # Initialize the sensors and tare the weight cell
    niclaVision = NiclaVision.NiclaVision()
    niclaVision.initialize(comPort="COM4")

    oakd_lite = poseCamera.OAKd_Camera()
    oakd_lite.enablePoseEstimation(
                pathCalMatrix = "E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/calibration/calibration_matrix.npy",
                pathDistCoeff = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/calibration/distortion_coefficients.npy',
                aruco_dict_type = 'DICT_7X7_50'
            )

    ftSensor = FTsensor.FTSensor()
    ftSensor.initialize()

    time.sleep(3)
    startTime = time.time()

    dataFrame = pd.DataFrame()
    samples = 0
    print("ready")
    # Read and store data
    while True:
        width=800
        niclaFrame = niclaVision.getFrame(imgShape = (30,45), imgType="sensor.GRAYSCALE", imgSize="sensor.HQVGA")
        print(niclaFrame)
        if niclaFrame is not None:
            h, w = niclaFrame.shape
            height = int(width*(h/w))
            niclaFramePreview = cv2.resize(niclaFrame, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("nicla_vision", niclaFramePreview)

        poseFrame = oakd_lite.getPoseFrame()
        h, w, _ = poseFrame.shape
        height = int(width*(h/w))
        poseFrame = cv2.resize(poseFrame, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("oakd_lite",poseFrame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            record = True
            print("recording")
        if key == ord('p'):
            record = False
            print("pause recording")
        if key == ord('q'):
            print("Exiting program")
            break

        if record:
            timeStamp = int(time.time() * 1000)
            forceDict = ftSensor.read(decimals=4)
            numberPoseTags = oakd_lite.getNumberTags()
            samples += 1
            dataFrame = utils.updateDataFrame(dataFrame, timeStamp, forceDict, numberPoseTags)
            
            utils.saveImage(niclaFrame, timeStamp, outputPath, cameraDevice="NiclaVision")
            utils.saveImage(oakd_lite.getFrame(), timeStamp, outputPath, cameraDevice="Oakd_lite")
            if samples % 20 == 0:
                utils.saveDataFrame(dataFrame, outputPath)
                dataFrame = pd.DataFrame()

    ftSensor.close()
    utils.saveDataFrame(dataFrame, outputPath)



