import sensor, image, time, ustruct, audio
from pyb import USB_VCP, LED


# In the future, I should add the posibilty of a sliding window of about 70-100ms = (2-3 * DEFAULT_CAPTURE_SIZE )
# DEFAULT_CAPTURE_SIZE =~ 31ms of sound

redLED = LED(1)

usb = USB_VCP()


CHANNELS = 1
FREQUENCY = 16000
# 1 second of audio = Frequency * 2 * CHANNELS , we then round it up to nearest power of 2.
SAMPLE_SIZE = 32768


DEFAULT_CAPTURE_SIZE = 1024
captureBuffer = bytearray(SAMPLE_SIZE)
bufCounter = 0

sendData = False
sampleAudio = False





audio.init(channels=CHANNELS, frequency=FREQUENCY, gain_db=26, highpass=0.9883)


def audio_callback(buf):
    # NOTE: do Not call any function that allocates memory.
    global rawBuffer
    global bufCounter
    global sampleAudio
    global sendData

    # Only sample the audio on request
    if sampleAudio:
        endIndex = bufCounter + DEFAULT_CAPTURE_SIZE 
        # If we reached the end, stop the sampling and enable sendData flag
        if (endIndex >= SAMPLE_SIZE):
            redLED.off()
            sampleAudio = False
            sendData = True

        captureBuffer[bufCounter: endIndex] = buf
        bufCounter += DEFAULT_CAPTURE_SIZE

    



audio.start_streaming(audio_callback)

while(True):
    cmd = usb.recv(4, timeout=5000)
    if (cmd == b'star'):
        # Start audio streaming
        
        sampleAudio = True
        bufCounter = 0
        redLED.on()
        


    elif (cmd == b'quit'):
        redLED.off()
        sendData = False
        audio.stop_streaming()
        break

    # Send data flag used to send a sample of data
    if sendData:
        usb.send(ustruct.pack("<L", len(captureBuffer)))
        usb.send(captureBuffer)
        sendData = False
        
