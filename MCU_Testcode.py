# Untitled - By: aleks - ons. des 14 2022

import sensor, image, time, os, tf, pyb, omv, ustruct, audio, math, utime
from pyb import LED
from ulab import numpy as np
from ulab import utils

import gc
import micropython
gc.collect()
micropython.mem_info()
print('-----------------------------')
print('Initial free: {} allocated: {}'.format(gc.mem_free(), gc.mem_alloc()))


# Setup for camera

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.HQVGA)      # Set frame size to QVGA (320x240)
sensor.skip_frames(time=2000)          # Let the camera adjust.

# Loading data for force estimation

maxFx = 75.1
maxFy = 79.6
maxFz = 133.0
minFx = -66.4
minFy = -91.8
minFz = 2.3
scaledOut = [0,0,0]

#net = tf.load("Model_30x45_quant.tflite", load_to_fb=True)
force_net = tf.load("Model_30x45_0.tflite", load_to_fb=True)


# Setup for Audio

CHANNELS = 1
FREQUENCY = 16000
# 1 second of audio = Frequency * 2 * CHANNELS , we then round it up to nearest power of 2.
SAMPLE_SIZE = 32768

DEFAULT_CAPTURE_SIZE = 1024
captureBuffer = bytearray(SAMPLE_SIZE)
bufCounter = 0

runTest = False
sampleAudio = False

# Extract MFCC features

numMfccCoeffs = 13
frameSize = 2048
hopSize = 512
numFilters = 26
minFreq = 0
maxFreq = 8000.0
numMelBands = 128
PI = 3.14159265358979323846


# Load data for Audio classification

audio_net = tf.load("audio_DNN16_16.tflite", load_to_fb=True)
audio.init(channels=CHANNELS, frequency=FREQUENCY, gain_db=26, highpass=0.9883)

def audio_callback(buf):
    # NOTE: do Not call any function that allocates memory.
    global bufCounter
    global sampleAudio
    global runTest

    # Only sample the audio on request
    if sampleAudio:
        endIndex = bufCounter + DEFAULT_CAPTURE_SIZE
        # If we reached the end, stop the sampling and enable runTest flag
        if (endIndex >= SAMPLE_SIZE):
            sampleAudio = False
            runTest = True

        captureBuffer[bufCounter: endIndex] = buf
        bufCounter += DEFAULT_CAPTURE_SIZE


def isolate_impact(audioData):
    startIndex = 0
    endIndex = len(audioData)

    for i in range(endIndex):
    # 0.025 in the range -1 to 1 => 6 in range -255 to 255
        if abs(audioData[i]) > 6:
            startIndex = i - 200
            if startIndex < 0:
                 startIndex = 0
            endIndex = startIndex + 1024
            break

    return audioData[startIndex:endIndex]


# Compute the Mel filterbank
def compute_mel_filterbank(num_filters, frame_size, sample_rate, min_freq, max_freq):
    mel_filterbank = []
    mel_min = 2595 * math.log10(1 + min_freq / 700.0)
    mel_max = 2595 * math.log10(1 + max_freq / 700.0)
    delta_mel = (mel_max - mel_min) / (num_filters + 1)

    for i in range(1, num_filters + 1):
        mel_center = mel_min + i * delta_mel
        freq_center = 700 * (math.pow(10, mel_center / 2595.0) - 1)

        bin_start = int(math.floor((frame_size + 1) * freq_center / sample_rate))
        bin_peak = int(math.floor((frame_size + 1) * freq_center / sample_rate))
        bin_end = int(math.ceil((frame_size + 1) * freq_center / sample_rate))

        filter_bank = [0.0] * (frame_size // 2 + 1)
        for k in range(bin_start, bin_end + 1):
            if k < bin_peak:
                filter_bank[k] = (k - bin_start) / (bin_peak - bin_start)
            else:
                filter_bank[k] = 1 - (k - bin_peak) / (bin_end - bin_peak)

        mel_filterbank.append(filter_bank)

    return mel_filterbank


# Compute the Discrete Cosine Transform (DCT) matrix
def compute_dct_matrix(num_coeffs, num_filters):
    dct_matrix = []
    sqrt_inv_n = 1 / math.sqrt(num_filters)

    for i in range(num_coeffs):
        row = []
        for j in range(num_filters):
            val = math.cos((PI * i * (2 * j + 1)) / (2 * num_filters)) * sqrt_inv_n
            row.append(val)

        dct_matrix.append(row)

    return dct_matrix


# Compute the MFCCs from the audio signal
def compute_mfcc(audio_signal, sample_rate, frame_size, hop_size, num_filters, num_coeffs, min_freq, max_freq):
    # Compute the Mel filterbank
    mel_filterbank = compute_mel_filterbank(num_filters, frame_size, sample_rate, min_freq, max_freq)

    # Compute the Discrete Cosine Transform (DCT) matrix
    dct_matrix = compute_dct_matrix(num_coeffs, num_filters)

    num_frames = ((len(audio_signal) - frame_size) // hop_size) + 1
    mfccs = []

    for i in range(num_frames):

        frame_start = i * hop_size
        frame_end = frame_start + frame_size
        frame = audio_signal[frame_start:frame_end]

        # Apply the Mel filterbank
        spectrum = abs(utils.spectrogram(frame))
        mel_filterbank = np.array(mel_filterbank)
        mel_energies = [0.0] * num_filters

        # Compute the mel energies using ulab numpy
        mel_energies = np.dot(mel_filterbank, spectrum[:frame_size // 2 + 1])

        # Ensure all values are non-negative (replace negative values with 0)
        mel_energies = np.maximum(mel_energies, 0.0)

        # Convert the mel_energies to a regular Python list
        mel_energies = mel_energies.tolist()



        # Take the logarithm of the Mel energies
        log_mel_energies = [math.log(mel_energy) for mel_energy in mel_energies]

        # Apply the Discrete Cosine

        mfcc = []
        for i in range(num_coeffs):
            mfcc_coeff = 0.0
            for j in range(num_filters):
                mfcc_coeff += log_mel_energies[j] * dct_matrix[i][j]
            mfcc.append(mfcc_coeff)

        mfccs.append(mfcc)

    mfccs = np.array(mfccs)


    return np.mean(mfccs, axis=0)









gc.collect()
print('Everything intilized and loaded: {} allocated: {}'.format(gc.mem_free(), gc.mem_alloc()))

#micropython.mem_info(1)



# Main
audio.start_streaming(audio_callback)
utime.sleep_ms(1500)
inferenceStart = time.ticks_ms()



clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()
    scaled_img = img.scale(x_size = 45, y_size = 30)
    #scaled_array = bytearray(scaled_img.size())



    #for i in range(100):
    output = force_net.classify(scaled_img)
    #print(output)


    scaledOut[0] = output[0][4][0] * (maxFx - minFx) + minFx
    scaledOut[1] = output[0][4][1] * (maxFy - minFy) + minFy
    scaledOut[2] = output[0][4][2] * (maxFz - minFz) + minFz

    #print(scaledOut)

    gc.collect()
    print('Image snapped and classified: {} allocated: {}'.format(gc.mem_free(), gc.mem_alloc()))

    inferenceDelta = time.ticks_add(time.ticks_ms(),-inferenceStart)
    #print("Inference speed (us): ",inferenceDelta)
    if inferenceDelta >= 5000:
        inferenceStart = time.ticks_ms()
        sampleAudio = True

    # Send data flag used to send a sample of data
    if runTest:
        captureBuffer = np.frombuffer(captureBuffer, dtype=np.int16)
        data = isolate_impact(captureBuffer)

        mfccs = compute_mfcc(data, FREQUENCY, frame_size=512, hop_size=160, num_filters=20, num_coeffs=13, min_freq=0, max_freq=8000.0)

        # Time the funcitons
        # start_time = utime.ticks_ms()

        for i in range(1):
            output = audio_net.regression(mfccs)
        print('Sound snapped and classified: {} allocated: {}'.format(gc.mem_free(), gc.mem_alloc()))
        #print(output)
        runTest = False
        # end_time = utime.ticks_ms()
        # elapsed_time = utime.ticks_diff(end_time, start_time)
        # print("Elapsed time: {} ms".format(elapsed_time))
    #print(clock.fps(), "fps")
