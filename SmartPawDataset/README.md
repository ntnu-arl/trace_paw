# Smart_paws dataset
This folder contains the scripts for collecting, synchronizing and cleaning the data from the Nicla Vision, force sensor and pose estimation camera. Aswell as scripts for gathering sound samples for terrain recognition. 

## How to collect force data
To collect data, you run the command **python dataCollection.py** 

Upload the device code under "sensors/NiclaVision/device_code/image/main.py" 

Before running the command you need to set the following variables:

    -   outputPath
    -   Nicla Vision com port
    -   pathCalMatrix
    -   pathDistCoeff
    -   aruco_dict_type (i.e. DICT_5X5_100)

When its running you can controll it with buttonpresses:

    'r' - start recording
    'p' - pause recording
    'q' - exit program


To clean the data, you will have to run the command **python dataCleaning.py**, this will run a series of cleaning tasks and leaving new .csv files for each step.

1. Remove all images where two aruco markers are not visible
2. Extract the relative orientation between the paw and the platform, and correct the force vector.
3. Normalize the force output between [0 , 1] 
 


## How to collect sound data
To collect data, you run the command **python soundCollection.py** 

Before running the command you need to do the following:

    -   Upload the device code under "sensors/NiclaVision/device_code/sound/main.py" 
    -   set outputPath
    -   set Nicla Vision com port
    -   set the desired terrain classes

When its running you can controll it with buttonpresses:

    'r' - start recording
    'p' - pause recording
    'q' - exit program


