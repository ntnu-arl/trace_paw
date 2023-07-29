import sensors.NiclaVision.usb_vcp_mic_transfer as NiclaVision
import utils as utils
import os
import pandas as pd
import time
import cv2


############################################################################

# This script gathers microphone data from the Nicla Vision and stores them in the output folder

############################################################################


if __name__ == '__main__':
    record = False

    print("---------------------------------------")
    print("Valid ground types: snow - grass - leafs - soil - rocks - solid -indoors")
    print("---------------------------------------")
    print("Enter the ground type you are sampling:")
    print("---------------------------------------")


    groundTypes = ["snow", "grass", "leafs", "soil", "rocks", "solid", "indoors"]
    groundType = ""
    while True:
        groundType = input()
        print("Your input: ", groundType)
        if groundType in groundTypes:
            break       
        else:
            print("Invalid input")
            





    outputPath = f'E:/Bruker/Dokumenter/Skole/Master/smart_paws/data/smallpaw_1/sound/{groundType}'
    os.makedirs(outputPath, exist_ok=True)

    # Initialize the sensors and misc
    niclaVision = NiclaVision.NiclaVision()
    niclaVision.initialize(comPort="COM3")

    # time.sleep(1)
    startTime = time.time()


    # Read and store data
    cv2.namedWindow("dummy")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            record = True
            print("recording one sample")
        if key == ord('p'):
            record = False
            print("pause recording")
        if key == ord('q'):
            print("Exiting program")
            niclaVision.quit()
            break

        if record:
            timeStamp = int(time.time() * 1000)
            niclaVision.startRecoding()
            soundArray = niclaVision.getSample()
            niclaVision.displaySample()
            record = False

            utils.saveSound(sound = soundArray, 
                            timeStamp = timeStamp, 
                            outputPath = outputPath, 
                            samplerate=niclaVision.sampleRate)
            
    



