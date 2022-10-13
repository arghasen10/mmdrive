import numpy as np
from scipy.signal import hilbert
import math
import cmath
from scipy.signal import correlate

# Commented out IPython magic to ensure Python compatibility.


# %cd /content/drive/MyDrive/SmileDetection

class SignalProcessor:
    # Constants
    SAMPLE_LENGTH = 2048
    NUMBER_OF_INITIAL = 5
    NUMBER_OF_SAMPLE_POINTS = 15

    def __init__(self):
        self.mixedSamplePhase = np.zeros((self.NUMBER_OF_INITIAL, self.SAMPLE_LENGTH))
        self.initialPoints = np.zeros((self.NUMBER_OF_SAMPLE_POINTS, 2))
        self.distances = np.zeros(self.NUMBER_OF_SAMPLE_POINTS)
        self.center = np.array([0., 0.])

    def signalMultiplication(self, x, y):
        out = correlate(x, y, mode='valid')

        maxAt = 0
        for i in range(len(out)):
            maxAt = i if out[i] > out[maxAt] else maxAt

        sample = np.zeros(len(x) - maxAt)
        for i in range(maxAt, len(x)):
            sample[i - maxAt] = x[i]

        return sample

    def hilbertTransform(self, x):
        h = hilbert(x)
        real, imag = np.real(h), np.imag(h)

        N = len(real)
        res = np.zeros((N, 2))
        for i in range(N):
            res[i] = [real[i], imag[i]]

        return res

    def dft(self, signal):
        N = len(signal)
        out = []
        for k in range(N):
            real, imag = 0., 0.
            for t in range(N):
                angle = (2 * math.pi * t * k) / N
                real += signal[t] * math.cos(angle)
                imag += -signal[t] * math.sin(angle)
            out.append(complex(real, imag))

        return np.array(out)

    def fourierTransform(self, chirp, direct, record):
        rSample = self.signalMultiplication(chirp, record)
        dSample = self.signalMultiplication(chirp, direct)

        analyticRSample = self.hilbertTransform(rSample)
        analyticDSample = self.hilbertTransform(dSample)
        analyticDirect = self.hilbertTransform(direct)
        analyticRecord = self.hilbertTransform(record)

        mixedRSignal = np.zeros(self.SAMPLE_LENGTH)
        mixedDSignal = np.zeros(self.SAMPLE_LENGTH)
        for i in range(self.SAMPLE_LENGTH):
            mixedRSignal[i] = np.dot(analyticRecord[i], analyticRSample[i])
            mixedDSignal[i] = np.dot(analyticDirect[i], analyticDSample[i])

        mixedRFftComplex = self.dft(mixedRSignal)
        mixedDFftcomplex = self.dft(mixedDSignal)

        res = mixedRFftComplex - mixedDFftcomplex
        amplitude = np.array([np.absolute(cmp) for cmp in res])
        phase = np.array([cmath.phase(cmp) for cmp in res])

        return np.array([amplitude, phase])


class data:
    def __init__(
            self,
            str_datetime,
            str_chirp,
            str_direct,
            str_record
    ):
        self.datetime = str_datetime
        self.chirp = self.parse(str_chirp.split(':')[1])
        self.direct = self.parse(str_direct.split(':')[1])
        self.record = self.parse(str_record.split(':')[1])

    def parse(self, str_arr):
        str_arr = str_arr.strip()
        str_arr = str_arr[1:]
        str_arr = str_arr[:-1]
        return np.array([float(num) for num in str_arr.split(',')])


data_nodes = []
with open('~/Documents/test_data_android/RawDataCollection.txt', 'r') as raw_data:
    counter = 0
    arr = []
    for line in raw_data:
        if len(arr) == 4:
            data_nodes.append(
                data(arr[0], arr[1], arr[2], arr[3])
            )
            arr = []
        arr.append(line)

    if len(arr) == 4:
        data_nodes.append(
            data(arr[0], arr[1], arr[2], arr[3])
        )
data_nodes = np.array(data_nodes)

data = data_nodes[0]
signalprocessor = SignalProcessor()
processed = signalprocessor.fourierTransform(data.chirp, data.record, data.direct)
print(processed[0].shape[0])
print(processed[0].mean())
print(processed[1].mean())
# print(data.chirp.shape[0])
# print(data.chirp.mean())
# print(data.direct.shape[0])
# print(data.direct.mean())
# print(data.record.shape[0])
# print(data.record.mean())
