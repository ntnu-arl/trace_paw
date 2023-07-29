# Image Transfer - As The Controller Device
#
# This script is meant to talk to the "image_transfer_jpg_as_the_remote_device_for_your_computer.py" on the OpenMV Cam.
#
# This script shows off how to transfer the frame buffer to your computer as a jpeg image.

import serial, serial.tools.list_ports, struct, sys
import sensors.NiclaVision.rpc as rpc
import cv2
import numpy as np

# Fix Python 2.x.
try: input = raw_input
except NameError: pass

# The RPC library above is installed on your OpenMV Cam and provides mutliple classes for
# allowing your OpenMV Cam to control over USB or WIFI.




##############################################################
# Call Back Handlers
##############################################################

        # You may change the pixformat and the framesize of the image transfered from the remote device
        # by modifying the below arguments.
        #
        # When cutthrough is False the image will be transferred through the RPC library with CRC and
        # retry protection on all data moved. For faster data transfer set cutthrough to True so that
        # get_bytes() and put_bytes() are called after an RPC call completes to transfer data
        # more quicly from one image buffer to another. Note: This works because once an RPC call
        # completes successfully both the master and slave devices are synchronized completely.
        #

def get_frame_buffer_call_back(interface ,pixformat_str, framesize_str, cutthrough, silent):
    if not silent: print("Getting Remote Frame...")

    result = interface.call("jpeg_image_snapshot", "%s,%s" % (pixformat_str, framesize_str))
    if result is not None:

        size = struct.unpack("<I", result)[0]
        img = bytearray(size)

        if cutthrough:
            # Fast cutthrough data transfer with no error checking.

            # Before starting the cut through data transfer we need to sync both the master and the
            # slave device. On return both devices are in sync.
            result = interface.call("jpeg_image_read")
            if result is not None:

                # GET BYTES NEEDS TO EXECUTE NEXT IMMEDIATELY WITH LITTLE DELAY NEXT.

                # Read all the image data in one very large transfer.
                interface.get_bytes(img, 5000) # timeout

        else:
            # Slower data transfer with error checking.

            # Transfer 32 KB chunks.
            chunk_size = (1 << 15)

            if not silent: print("Reading %d bytes..." % size)
            for i in range(0, size, chunk_size):
                ok = False
                for j in range(3): # Try up to 3 times.
                    result = interface.call("jpeg_image_read", struct.pack("<II", i, chunk_size))
                    if result is not None:
                        img[i:i+chunk_size] = result # Write the image data.
                        if not silent: print("%.2f%%" % ((i * 100) / size))
                        ok = True
                        break
                    if not silent: print("Retrying... %d/2" % (j + 1))
                if not ok:
                    if not silent: print("Error!")
                    return None

        return img

    else:
        if not silent: print("Failed to get Remote Frame!")

    return None










class NiclaVision:
    def __init__(self): 
        self.interface = None
        self.frame = None

    def initialize(self, comPort):
        """
        Method for initializing the communcation interface with the NiclaVision

        Args:
            comPort (string) : example "COM4"
        """

        print("\nAvailable Ports:\n")
        for port, desc, hwid in serial.tools.list_ports.comports():
            print("{} : {} [{}]".format(port, desc, hwid))
        # sys.stdout.write("\nPlease enter a port name: ")
        sys.stdout.flush()
        self.interface = rpc.rpc_usb_vcp_master(port=comPort)
        print("")
        sys.stdout.flush()
        


    def getFrame(self, imgShape=(30,45) ,imgType="sensor.GRAYSCALE", imgSize="sensor.QVGA"):
        """
        Method for getting the image frame from the niclaVision

        Args:
            imgType (string) : set the image type to use
            imgSize (string) : set the image resolution

        Returns:
            frame (numpy.ndarray) : frame containing the image 
        """
        sys.stdout.flush()
        img = get_frame_buffer_call_back(self.interface, imgType, imgSize, cutthrough=True, silent=True)
        if img is not None:
                image = np.asarray(img)
                image = image.reshape(imgShape)
                # print(image)
                # image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                self.frame = image
        return self.frame
