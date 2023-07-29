import sys, serial, struct, time
import serial.tools.list_ports
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt
import pandas as pd


class NiclaVision:
    def __init__(self): 
        self.interface = None
        self.audioSample = None
        self.sampleRate = 16000
        self.maxSampleSize = 32768
        self.sampleCounter = 0

    def initialize(self, comPort="COM3"):
        """
        Method for initializing the communcation interface with the NiclaVision

        Args:
            comPort (string) : example "COM3"
        """

        print("\nAvailable Ports:\n")
        for port, desc, hwid in serial.tools.list_ports.comports():
            print("{} : {} [{}]".format(port, desc, hwid))
        # sys.stdout.write("\nPlease enter a port name: ")
        sys.stdout.flush()


        self.interface = serial.Serial(comPort, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                    xonxoff=False, rtscts=False, stopbits=serial.STOPBITS_ONE, timeout=1, dsrdtr=True)
        self.interface.setDTR(True) # dsrdtr is ignored on Windows.

        self.interface.flush()
        


    def startRecoding(self, timeFrame="1000"):
        print("sending start")
        self.audioSample = np.array([])
        self.sampleCounter = 0
        self.interface.flush()
        self.interface.write(b'star')
        self.interface.flush()



    def getSample(self):
        while (not self.interface.inWaiting()):
                pass
        size = struct.unpack('<L', self.interface.read(4))[0]
        print("Sample size: ", size)
        captureBuffer = self.interface.read(size)
        self.audioSample = np.frombuffer(captureBuffer, dtype=np.int16)
        print(self.audioSample)

        return self.audioSample


    def displaySample(self):
        pd.Series(self.audioSample).plot(figsize=(10,5), 
                    lw=1,
                    title='Raw audio example')

        plt.show()

        play_obj = sa.play_buffer(self.audioSample, num_channels=1, bytes_per_sample=2, sample_rate=self.sampleRate)

        play_obj.wait_done()

    def quit(self):
        self.interface.flush()
        self.interface.write(b'quit')
        self.interface.flush()





if __name__ == '__main__':
    nicla = NiclaVision()

    nicla.initialize("COM3")

    # time.sleep(1)
    nicla.startRecoding()

    nicla.getSample()
    nicla.displaySample()
    print("shutting down")


