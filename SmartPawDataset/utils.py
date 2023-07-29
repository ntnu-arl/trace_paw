import pandas as pd
import os
import cv2
import scipy.io.wavfile

def updateDataFrame(dataFrame, timeStamp, FTsensorDict=None, numberPoseTags=None):
    """
    Method for updating the Pandas data frame

    Args:
        dataFrame (pd.DataFrame): Pandas data frame
        timeStamp (int)         : UNIX time in milliseconds
        FTsensorDict (dict)     : Force and torque data from FT sensor
        PoseDict    (dict)      : Pose of the paw

    Returns:
        dataFrame (pd.DataFrame): Pandas data frame 
    """
    data = {
    "TimeStamp" : timeStamp,
    "IDs": numberPoseTags,
    }

    for key in FTsensorDict:
        data[key] = FTsensorDict[key]

    df2 = pd.DataFrame(data, index=[0])
    return pd.concat([dataFrame, df2])


def saveDataFrame(dataFrame, outputPath, filetype = 'CSV', name = 'metadata'):
    """
    Method for saving the Pandas data frame to CSV file or pikle file depending on the input.

    Args:
        dataFrame (pd.DataFrame): Pandas data frame
        outputPath              : folder path for output
        filetype                : string containing file type, pickle or CSV 
        name                    : string containing the file name, default = metadata
    """
    os.makedirs(outputPath, exist_ok=True)
    
    if filetype == 'pickle':
        filename = outputPath + name + ".pkl"
        if os.path.exists(filename):
            dataFrame.to_pickle(filename, mode='a')
        else:
            dataFrame.to_pickle(filename)
 
    else:
        filename = outputPath + name + ".csv"
        if os.path.exists(filename):    
            dataFrame.to_csv(filename, mode='a', index=False, header=False)
        else:
            dataFrame.to_csv(filename, index=False)


def saveImage(image, timeStamp, outputPath, cameraDevice='NiclaVision'):
    """
    Method for updating an image to jpeg format.

    Args:
        image (numpy.ndarray)   : frame containing the image 
        timeStamp (int)         : UNIX time in milliseconds
        cameraDevice (string)   : device name
    """
    path = f'{outputPath}{cameraDevice}/'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f'{path}{str(timeStamp)}.jpeg',image)


def saveSound(sound, timeStamp, outputPath, samplerate=16000):
    """
    Method for updating an sound sample to wav format.

    Args:
        sound (numpy.ndarray)   : frame containing the sound clip 
        timeStamp (int)         : UNIX time in milliseconds
        outputPAth              : string containing output path
        samplerate              : int containing sampling rate for sound clip
    """
    path = f'{outputPath}/'
    os.makedirs(path, exist_ok=True)
    scipy.io.wavfile.write(filename = f'{path}{str(timeStamp)}.wav',
                           rate = samplerate,
                           data = sound)



