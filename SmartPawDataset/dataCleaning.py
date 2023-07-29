import pandas as pd
import numpy as np
import os
from sensors.Oakd_Lite.aruco_pose import aruco_pose
import cv2
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation as R

# Set up the file paths to our data set 
filepath = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/data/machined/'
filepathAruco = filepath + 'Oakd_lite/'
filepathNiclaVision = filepath + 'NiclaVision/'

# Filepath to the calibration files
pathCalMatrix = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/calibration/calibration_matrix.npy'
pathDistCoeff = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/SmartPawDataset/sensors/Oakd_Lite/calibration/distortion_coefficients.npy'

# Initialize the aruco_pose_estimator
aruco_estimation = aruco_pose(
               pathCalMatrix = pathCalMatrix,
                pathDistCoeff = pathDistCoeff)



outputfile = 'metadata_1.csv'
if not os.path.exists(filepath + outputfile):
    df = pd.read_csv(filepath + 'metadata.csv')
    print(df.head())
    df2 = df.copy()

    # Fix the axis of the force sensor wrt. the testbed orientation (flipping about Y-axis)
    df2["Fx"] = -df['Fx']
    # df2["Fy"] = df['Fx']
    df2["Fz"] = -df["Fz"]
    df2 = df2.drop(columns = ['IDs', 'Tx', 'Ty', 'Tz'])


    for name in tqdm(df2.loc[:]["TimeStamp"]):
        index = df.index[df['TimeStamp'] == name]
        file = str(name) + '.jpeg'
        arucoFile = os.path.join(filepathAruco, file)
        arucoImage = cv2.imread(arucoFile)
        patternFile = os.path.join(filepathNiclaVision, file)
        patternImage = cv2.imread(patternFile)

        # Check if the image exists
        if isinstance(arucoImage, np.ndarray) and isinstance(patternImage, np.ndarray):
            # Step 1: Find the images with both aruco tags visible
            if [df.loc[index]['IDs'] == 2]: 
                #Step 2: Get the pose of the aruco markers in the image
                if name >= 1683453705609 and name <= 1683456648336:
                    poseDict = aruco_estimation.estimateArucoPose(arucoImage, extraRot = True)
                else:
                    poseDict = aruco_estimation.estimateArucoPose(arucoImage)
                    
                relativeRotVec = aruco_estimation.estimateRelativePose(poseDict) # RelativeRotVec is euler vector of mode zyx
                if relativeRotVec is not None: 
                    df2.loc[index, 'yaw'] = relativeRotVec[0]
                    df2.loc[index, 'pitch'] = relativeRotVec[1]
                    df2.loc[index, 'roll'] = relativeRotVec[2]

    df2 = df2.dropna()
    print(df2)
    filename =  filepath + outputfile
    df2.to_csv(filename, index=False)



# Step 3: Copy the images of the new sliced dataset
newfilePathAruco = filepath + 'cleanData/Oakd_lite/'
newfilePathNiclaVision =  filepath + 'cleanData/NiclaVision/'
if not os.path.exists(newfilePathAruco):
    os.makedirs(newfilePathAruco, exist_ok=True)
    os.makedirs(newfilePathNiclaVision, exist_ok=True)

    df = pd.read_csv(filepath + outputfile)
    for name in tqdm(df.loc[:]["TimeStamp"]):
        image = str(name) + '.jpeg'
        fileAruco = os.path.join(filepathAruco, image)
        newfileAruco = os.path.join(newfilePathAruco, image)
        shutil.copy(fileAruco, newfileAruco)
        # shutil.move(fileAruco, newfileAruco)
        fileNiclaVision = os.path.join(filepathNiclaVision, image)
        newfileNiclaVision = os.path.join(newfilePathNiclaVision, image)
        shutil.copy(fileNiclaVision, newfileNiclaVision)




# Step 4: Correct the force vector wrt to relative pose
outputfile = 'metadata_2.csv'
if not os.path.exists(filepath + outputfile):
    df = pd.read_csv(filepath + 'metadata_1.csv')
    df2 = df.copy()

    dfLen = df.shape[0]
    for i in tqdm(range(0, dfLen)):
        rvec_euler = np.array([df.loc[i, 'yaw'], df.loc[i, 'pitch'], df.loc[i, 'roll']])
        rvec = R.from_euler('zyx', rvec_euler)
        # print("rvec" , rvec_euler)
        rot_matrix = R.as_matrix(rvec)
        # print(rot_matrix)
        force_vec = np.array([df.loc[i, 'Fx'], df.loc[i, 'Fy'], df.loc[i, 'Fz']])
        # force_vec = np.array([0, 0, np.linalg.norm(force_vec)])
        # force_vec2 = force_vec @ rot_matrix
        force_vec2 = rot_matrix @ force_vec
        # print("Force vec:", force_vec, force_vec2)
        df2.loc[i, 'Fx'] = force_vec2[0]
        df2.loc[i, 'Fy'] = force_vec2[1]
        if force_vec2[2] < 0:
            df2.loc[i, 'Fz'] = 0
        else:
            df2.loc[i, 'Fz'] = force_vec2[2]

        # print("timestamp", df2.loc[i, "TimeStamp"])
    print(df2)
    filename =  filepath + outputfile
    df2.to_csv(filename, index=False)


# Step 5: Normalize the Force output between 0 and 1
outputfile = 'metadata_normalized.csv'
if not os.path.exists(filepath + outputfile):
    df = pd.read_csv(filepath + 'metadata_2.csv')
    df2 = df.copy()

    for feature_name in df.columns:
        forces = ['Fx', 'Fy', 'Fz']
        if feature_name in forces:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            df2[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    print(df2)
    filename =  filepath + outputfile
    df2.to_csv(filename, index=False)




