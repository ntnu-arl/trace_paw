import nidaqmx
import numpy as np
import xml.etree.ElementTree as ET


class FTSensor:
    def __init__(self):
        """This class will handle the connection, reading and correcting,
        of the sensor data from the Force Torque Sensor mini45
        
        """
        self.CalData = np.ones((6,6))
        self.Bias = np.zeros(6)
        self.sensorName = ''
        self.task = None

    def initialize(self, sensorName='mini45', calibFile='./sensors/ForceTorqueSensor/FT38684.cal'):
        """Method for initialization of the sensor. 
        It starts the sensor reading task, 
        loads the calibration matrix, 
        open the channels we wish to read and set the bias for the sensor
        
        Args:
            sensorName (str): the sensor name in NI MAX software
            calibFile (str): location of the calibration file
        """
        self.sensorName = sensorName
        self.task = nidaqmx.Task()
        self.loadCalibration(calibFile)

        # Channels correponds to: ai0: Fx, ai1: Fy, ai2: Fz
        channels = ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5']
        for aiX in channels:
            self.task.ai_channels.add_ai_voltage_chan(f'mini45/{aiX}')
        self.setBias()


    def loadCalibration(self, calibFile='./sensors/ForceTorqueSensor/FT38684.cal'):
        """Method for reading the calibration file and generating the calibration matrix

        Args:
            calibFile (str): location of the calibration file 
        """

        tree = ET.parse(calibFile)
        root = tree.getroot()
        calValues = []

        # Extract <UserAxis> from calibration file 
        for child in root[0]:
            if child.tag == 'UserAxis':

                # Convert values from string to Floats
                valueStr = child.attrib['values']
                values = []
                for item in valueStr.split(' '):
                    if not item.isspace() and len(item) >= 1:
                        values.append(float(item))
                calValues.append(values)

        self.CalData = np.array(calValues)
        print('Calibration matrix ready')



    def setBias(self):
        # Read out the raw sensor data
        self.task.start()
        rawValue = self.task.read()
        self.task.stop()
        self.Bias = np.array(rawValue)
        print('Bias is set')



    def read(self, decimals=4): # private
        """Method for reading the sensor data as Force (N)

        Args:
            decimals (int) : number of decimals in sensor value

        Returns:
            sensorData (Dict) : dictionary containing sensor data 
        """
        
        labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        sensorData = {}
        # Read out the raw sensor data
        self.task.start()
        rawValue = self.task.read()
        self.task.stop()
        rawValue = np.array(rawValue)
        # Correct the data for bias error
        biasCorrected = rawValue - self.Bias
        # Apply the calibarion matrix
        sensData = np.dot(self.CalData,biasCorrected.T)
        for i in range(len(labels)):
            sensorData[labels[i]] = float(round(sensData[i], decimals))
        return sensorData

    def close(self):
        self.task.close()

